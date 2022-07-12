/*
 * This implements parallel page copy function through multi threaded
 * work queues.
 *
 * Zi Yan <ziy@nvidia.com>
 *
 * This work is licensed under the terms of the GNU GPL, version 2.
 */
#include <linux/highmem.h>
#include <linux/workqueue.h>
#include <linux/slab.h>
#include <linux/freezer.h>

#include <linux/migrate.h>

#define _MM_MALLOC_H_INCLUDED 
#include <immintrin.h>
#undef _MM_MALLOC_H_INCLUDED 

/*
 * nr_copythreads can be the highest number of threads for given node
 * on any architecture. The actual number of copy threads will be
 * limited by the cpumask weight of the target node.
 */
extern unsigned int limit_mt_num;

struct copy_page_info {
	struct work_struct copy_page_work;
	char *to;
	char *from;
	unsigned long chunk_size;
};


int sysctl_enable_nt_exchange = 0;

__attribute__((optimize("-O3")))
__attribute__((target("avx512vl,bmi2")))
static void exchange_page_routine(char *to, char *from, unsigned long chunk_size)
{
#ifdef CONFIG_AS_AVX512
	if(sysctl_enable_nt_exchange==1){
		// use non-temporal load/stores
		__m512i_u* s = (__m512i_u*)from;
		__m512i_u* d = (__m512i_u*)to;
		__m512i_u temp;
		unsigned long i;
		
		for(i=0; i<chunk_size; i+=64){
			temp =  _mm512_stream_load_si512(s);
			_mm512_stream_si512(s, _mm512_stream_load_si512(d));
			_mm512_stream_si512(d, temp);
			s++, d++;
		} 
	} else {
		u64 tmp;
		int i;

		for (i = 0; i < chunk_size; i += sizeof(tmp)) {
			tmp = *((u64*)(from + i));
			*((u64*)(from + i)) = *((u64*)(to + i));
			*((u64*)(to + i)) = tmp;
		}
	}
#else
	u64 tmp;
	int i;

	for (i = 0; i < chunk_size; i += sizeof(tmp)) {
		tmp = *((u64*)(from + i));
		*((u64*)(from + i)) = *((u64*)(to + i));
		*((u64*)(to + i)) = tmp;
	}	
#endif
}

static void exchange_page_work_queue_thread(struct work_struct *work)
{
	struct copy_page_info *my_work = (struct copy_page_info*)work;
	kernel_fpu_begin();
	exchange_page_routine(my_work->to,
							  my_work->from,
							  my_work->chunk_size);
	kernel_fpu_end();
}

extern int sysctl_enable_page_migration_optimization_avoid_remote_pmem_write;

int exchange_page_mthread(struct page *to, struct page *from, int nr_pages)
{
	int total_mt_num = limit_mt_num;
	int to_node, from_node;

// #ifdef CONFIG_PAGE_MIGRATION_PROFILE
// 	int to_node = page_to_nid(to);
// #else
// 	int to_node = numa_node_id();
// #endif
	int i;
	struct copy_page_info *work_items;
	char *vto, *vfrom;
	unsigned long chunk_size;
	const struct cpumask *per_node_cpumask;
	int cpu_id_list[32] = {0};
	int cpu;

	from_node = page_to_nid(from);
	to_node = page_to_nid(to);

	per_node_cpumask = cpumask_of_node(numa_node_id());

	if(sysctl_enable_page_migration_optimization_avoid_remote_pmem_write){
		if(get_nearest_cpu_node(to_node)!=-1){
			// from_node is pmem only memory node
			per_node_cpumask = cpumask_of_node(cpu_to_node(get_nearest_cpu_node(to_node)));
		} else if(get_nearest_cpu_node(from_node)!=-1) {
			per_node_cpumask = cpumask_of_node(cpu_to_node(get_nearest_cpu_node(from_node)));
		}
	}

	total_mt_num = min_t(unsigned int, total_mt_num,
						 cpumask_weight(per_node_cpumask));

	if (total_mt_num > 1)
		total_mt_num = (total_mt_num / 2) * 2;

	if (total_mt_num > 32 || total_mt_num < 1)
		return -ENODEV;

	work_items = kvzalloc(sizeof(struct copy_page_info)*total_mt_num,
						 GFP_KERNEL);
	if (!work_items)
		return -ENOMEM;

	i = 0;
	for_each_cpu(cpu, per_node_cpumask) {
		if (i >= total_mt_num)
			break;
		cpu_id_list[i] = cpu;
		++i;
	}

	/* XXX: assume no highmem  */
	vfrom = kmap(from);
	vto = kmap(to);
	chunk_size = PAGE_SIZE*nr_pages / total_mt_num;

	for (i = 0; i < total_mt_num; ++i) {
		INIT_WORK((struct work_struct *)&work_items[i],
				exchange_page_work_queue_thread);

		work_items[i].to = vto + i * chunk_size;
		work_items[i].from = vfrom + i * chunk_size;
		work_items[i].chunk_size = chunk_size;

		queue_work_on(cpu_id_list[i],
					  system_highpri_wq,
					  (struct work_struct *)&work_items[i]);
	}

	/* Wait until it finishes  */
	flush_workqueue(system_highpri_wq);

	kunmap(to);
	kunmap(from);

	kvfree(work_items);

	return 0;
}

long long int total_base_pages_exchanged;
long long int total_time_taken_in_exchange;
int sysctl_reset_bandwidth_counters;

int exchange_page_lists_mthread(struct page **to, struct page **from, int nr_pages)
{	
	int err = 0;
	int total_mt_num = limit_mt_num;
	int to_node, from_node, node_selected_for_migration_processing;
	
// #ifdef CONFIG_PAGE_MIGRATION_PROFILE
// 	int to_node = page_to_nid(*to);
// #else
// 	int to_node = numa_node_id();
// #endif
	int i;
	struct copy_page_info *work_items;
	const struct cpumask *per_node_cpumask;
	int cpu_id_list[32] = {0};
	int cpu;
	int item_idx;

	long long int timestamp;
	long long int nr_pages_backup = nr_pages;
	timestamp = rdtsc();

	from_node = page_to_nid(*from);
	to_node = page_to_nid(*to);

	per_node_cpumask = cpumask_of_node(node_selected_for_migration_processing=numa_node_id());

	if(sysctl_enable_page_migration_optimization_avoid_remote_pmem_write){
		int from_node_cpu_count = cpumask_weight(cpumask_of_node(from_node));
		int to_node_cpu_count = cpumask_weight(cpumask_of_node(to_node));
		
		pr_debug("[exchange_page_lists_mthread-current_cpu_node=%d] PMEM optimization enabled: nr_pages=%d, limit_mt_num=%d, from_node_cpu_count=%d, to_node_cpu_count=%d\n", 
			numa_node_id(), nr_pages, limit_mt_num, from_node_cpu_count, to_node_cpu_count);
		pr_debug("[exchange_page_lists_mthread] PMEM optimization enabled: get_nearest_cpu_node(%d)=%d, get_nearest_cpu_node(%d)=%d\n", 
			from_node, get_nearest_cpu_node(from_node), to_node, get_nearest_cpu_node(to_node));
		
		if(get_nearest_cpu_node(to_node)!=-1){
			// from_node is pmem only memory node
			node_selected_for_migration_processing = cpu_to_node(get_nearest_cpu_node(to_node));
			per_node_cpumask = cpumask_of_node(node_selected_for_migration_processing);
		} else if(get_nearest_cpu_node(from_node)!=-1) {
			node_selected_for_migration_processing = cpu_to_node(get_nearest_cpu_node(from_node));
			per_node_cpumask = cpumask_of_node(node_selected_for_migration_processing);
		}
	}

	total_mt_num = min_t(unsigned int, total_mt_num,
						 cpumask_weight(per_node_cpumask));

	if (total_mt_num > 32 || total_mt_num < 1)
		return -ENODEV;

	if (nr_pages < total_mt_num) {
		int residual_nr_pages = nr_pages - rounddown_pow_of_two(nr_pages);

		if (residual_nr_pages) {
			for (i = 0; i < residual_nr_pages; ++i) {
				BUG_ON(hpage_nr_pages(to[i]) != hpage_nr_pages(from[i]));
				err = exchange_page_mthread(to[i], from[i], hpage_nr_pages(to[i]));
				VM_BUG_ON(err);
			}
			nr_pages = rounddown_pow_of_two(nr_pages);
			to = &to[residual_nr_pages];
			from = &from[residual_nr_pages];
		}

		work_items = kvzalloc(sizeof(struct copy_page_info)*total_mt_num,
							 GFP_KERNEL);
	} else
		work_items = kvzalloc(sizeof(struct copy_page_info)*nr_pages,
							 GFP_KERNEL);
	if (!work_items)
		return -ENOMEM;

	i = 0;
	for_each_cpu(cpu, per_node_cpumask) {
		if (i >= total_mt_num)
			break;
		cpu_id_list[i] = cpu;
		++i;
	}

	if (nr_pages < total_mt_num) {
		for (cpu = 0; cpu < total_mt_num; ++cpu)
			INIT_WORK((struct work_struct *)&work_items[cpu],
					  exchange_page_work_queue_thread);
		cpu = 0;
		for (item_idx = 0; item_idx < nr_pages; ++item_idx) {
			unsigned long chunk_size = nr_pages * PAGE_SIZE * hpage_nr_pages(from[item_idx]) / total_mt_num;
			char *vfrom = kmap(from[item_idx]);
			char *vto = kmap(to[item_idx]);
			VM_BUG_ON(PAGE_SIZE * hpage_nr_pages(from[item_idx]) % total_mt_num);
			VM_BUG_ON(total_mt_num % nr_pages);
			BUG_ON(hpage_nr_pages(to[item_idx]) !=
				   hpage_nr_pages(from[item_idx]));

			for (i = 0; i < (total_mt_num/nr_pages); ++cpu, ++i) {
				work_items[cpu].to = vto + chunk_size * i;
				work_items[cpu].from = vfrom + chunk_size * i;
				work_items[cpu].chunk_size = chunk_size;
			}
		}
		if (cpu != total_mt_num)
			pr_err("%s: only %d out of %d pages are transferred\n", __func__,
				cpu - 1, total_mt_num);

		for (cpu = 0; cpu < total_mt_num; ++cpu)
			queue_work_on(cpu_id_list[cpu],
						  system_highpri_wq,
						  (struct work_struct *)&work_items[cpu]);
	} else {
		for (i = 0; i < nr_pages; ++i) {
			int thread_idx = i % total_mt_num;

			INIT_WORK((struct work_struct *)&work_items[i], exchange_page_work_queue_thread);

			/* XXX: assume no highmem  */
			work_items[i].to = kmap(to[i]);
			work_items[i].from = kmap(from[i]);
			work_items[i].chunk_size = PAGE_SIZE * hpage_nr_pages(from[i]);

			BUG_ON(hpage_nr_pages(to[i]) != hpage_nr_pages(from[i]));

			queue_work_on(cpu_id_list[thread_idx], system_highpri_wq, (struct work_struct *)&work_items[i]);
		}
	}

	pr_debug("[exchange_page_lists_mthread-current_cpu_node=%d] Scheduled the mt-%d exchange operation between node %d and %d on CPU close to node %d!\n", numa_node_id(), total_mt_num, from_node, to_node, node_selected_for_migration_processing);

	/* Wait until it finishes  */
	flush_workqueue(system_highpri_wq);

	if(sysctl_reset_bandwidth_counters){
		total_base_pages_exchanged = 0;
		total_time_taken_in_exchange = 0;
		sysctl_reset_bandwidth_counters = 0;
	}
	total_base_pages_exchanged += hpage_nr_pages(to[0])*nr_pages_backup;
	total_time_taken_in_exchange += rdtsc() - timestamp;
	pr_debug("[exchange_page_lists_mthread] Current exchange bandhwidth: %d KB/GCycles\n", total_base_pages_exchanged*4*1000000000/total_time_taken_in_exchange);

	for (i = 0; i < nr_pages; ++i) {
			kunmap(to[i]);
			kunmap(from[i]);
	}

	kvfree(work_items);

	return err;
}

