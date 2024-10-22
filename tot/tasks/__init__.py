def get_task(name, args=None):
    if name == 'game24':
        from tot.tasks.game24 import Game24Task
        return Game24Task()
    elif name == 'knights_and_knaves':
        from tot.tasks.knights_and_knaves import Knights_and_Knaves_Task
        return Knights_and_Knaves_Task(args.n_char)
    else:
        raise NotImplementedError