# ProcessingTime.msg
string id       # node ID
builtin_interfaces/Time t_in    # processing started
builtin_interfaces/Time t_out   # processing ended
builtin_interfaces/Duration delta  # (t_out - t_in)
