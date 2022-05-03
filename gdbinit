file ./vmlinux
target remote localhost:1234
break do_move_pages_to_node
continue
