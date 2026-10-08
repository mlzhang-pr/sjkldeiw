





GOOD_RPO_ENVS = [
    "door-close-v2",
    "sweep-into-v2",
    "coffee-button-v2",
    "window-open-v2",
    "reach-wall-v2",
    "drawer-close-v2",
    "button-press-v2",
    "plate-slide-back-side-v2",
    "reach-v2",
    "plate-slide-side-v2",
    "coffee-push-v2",
    "plate-slide-back-v2",
    "soccer-v2",
    "window-close-v2",
    "handle-pull-side-v2",
    "hand-insert-v2",
    "door-lock-v2",
    "push-v2",
    "peg-unplug-side-v2"
]












RPO10_SHORT = [['door-close-v2','coffee-button-v2']]

RPO10_SEQ = [['handle-pull-side-v2', 'peg-unplug-side-v2', 'coffee-push-v2', 'soccer-v2', 'drawer-close-v2', 'reach-wall-v2', 'plate-slide-back-v2', 'window-open-v2', 'plate-slide-side-v2', 'plate-slide-back-side-v2'],
             ['window-close-v2', 'window-open-v2', 'hand-insert-v2', 'door-lock-v2', 'reach-v2', 'button-press-v2', 'sweep-into-v2', 'coffee-button-v2', 'door-close-v2', 'push-v2'],
             ['window-close-v2', 'reach-wall-v2', 'sweep-into-v2', 'reach-v2', 'soccer-v2', 'coffee-push-v2', 'plate-slide-side-v2', 'drawer-close-v2', 'hand-insert-v2', 'door-close-v2'],
             ['plate-slide-back-v2', 'reach-wall-v2', 'door-lock-v2', 'peg-unplug-side-v2', 'push-v2', 'button-press-v2', 'plate-slide-back-side-v2', 'coffee-push-v2', 'coffee-button-v2', 'handle-pull-side-v2'],
             ['push-v2', 'coffee-button-v2', 'sweep-into-v2', 'door-close-v2', 'drawer-close-v2', 'soccer-v2', 'peg-unplug-side-v2', 'hand-insert-v2', 'door-lock-v2', 'reach-v2'],
             ['button-press-v2', 'plate-slide-back-side-v2', 'window-close-v2', 'plate-slide-side-v2', 'peg-unplug-side-v2', 'plate-slide-back-v2', 'coffee-button-v2', 'window-open-v2', 'handle-pull-side-v2', 'door-close-v2'],
             ['push-v2', 'button-press-v2', 'plate-slide-back-v2', 'drawer-close-v2', 'soccer-v2', 'plate-slide-side-v2', 'reach-wall-v2', 'coffee-push-v2', 'window-close-v2', 'door-lock-v2'],
             ['plate-slide-side-v2', 'hand-insert-v2', 'handle-pull-side-v2', 'plate-slide-back-side-v2', 'window-open-v2', 'sweep-into-v2', 'reach-wall-v2', 'reach-v2', 'soccer-v2', 'peg-unplug-side-v2'],
             ['hand-insert-v2', 'reach-v2', 'window-close-v2', 'drawer-close-v2', 'window-open-v2', 'coffee-button-v2', 'plate-slide-back-v2', 'coffee-push-v2', 'push-v2', 'plate-slide-back-side-v2'],
             ['sweep-into-v2', 'peg-unplug-side-v2', 'window-close-v2', 'door-lock-v2', 'hand-insert-v2', 'handle-pull-side-v2', 'window-open-v2', 'door-close-v2', 'button-press-v2', 'reach-wall-v2'],
             ['reach-v2', 'door-lock-v2', 'sweep-into-v2', 'push-v2', 'button-press-v2', 'coffee-push-v2', 'handle-pull-side-v2', 'plate-slide-side-v2', 'door-close-v2', 'drawer-close-v2'],
             ['plate-slide-back-side-v2', 'soccer-v2', 'sweep-into-v2', 'handle-pull-side-v2', 'plate-slide-side-v2', 'peg-unplug-side-v2', 'door-lock-v2', 'reach-v2', 'plate-slide-back-v2', 'coffee-button-v2'],
             ['reach-wall-v2', 'plate-slide-back-v2', 'drawer-close-v2', 'hand-insert-v2', 'coffee-push-v2', 'coffee-button-v2', 'window-close-v2', 'plate-slide-back-side-v2', 'door-close-v2', 'button-press-v2'],
             ['soccer-v2', 'drawer-close-v2', 'push-v2', 'sweep-into-v2', 'window-open-v2', 'reach-wall-v2', 'door-lock-v2', 'window-close-v2', 'reach-v2', 'hand-insert-v2'],
             ['plate-slide-back-v2', 'plate-slide-side-v2', 'door-close-v2', 'push-v2', 'peg-unplug-side-v2', 'plate-slide-back-side-v2', 'coffee-push-v2', 'coffee-button-v2', 'button-press-v2', 'soccer-v2'],
             ['hand-insert-v2', 'coffee-button-v2', 'soccer-v2', 'window-open-v2', 'push-v2', 'reach-v2', 'drawer-close-v2', 'handle-pull-side-v2', 'door-lock-v2', 'plate-slide-back-side-v2'],
             ['coffee-push-v2', 'door-close-v2', 'handle-pull-side-v2', 'window-close-v2', 'plate-slide-back-v2', 'reach-wall-v2', 'sweep-into-v2', 'window-open-v2', 'plate-slide-side-v2', 'peg-unplug-side-v2'],
             ['coffee-push-v2', 'button-press-v2', 'reach-v2', 'peg-unplug-side-v2', 'reach-wall-v2', 'door-close-v2', 'window-open-v2', 'handle-pull-side-v2', 'plate-slide-back-side-v2', 'soccer-v2'],
             ['sweep-into-v2', 'plate-slide-side-v2', 'button-press-v2', 'drawer-close-v2', 'push-v2', 'coffee-button-v2', 'door-lock-v2', 'hand-insert-v2', 'plate-slide-back-v2', 'window-close-v2'],
             ['reach-v2', 'button-press-v2', 'plate-slide-side-v2', 'door-close-v2', 'plate-slide-back-side-v2', 'plate-slide-back-v2', 'coffee-button-v2', 'sweep-into-v2', 'reach-wall-v2', 'drawer-close-v2'],
             ['button-press-v2', 'plate-slide-back-side-v2', 'window-close-v2'],]


RPO10_SEQ_OLD = [['plate-slide-back-v2', 'button-press-v2', 'handle-pull-side-v2', 'peg-unplug-side-v2', 'window-open-v2', 'handle-pull-v2', 'plate-slide-back-side-v2', 'coffee-push-v2', 'push-v2', 'reach-v2'],
             ['door-close-v2', 'hand-insert-v2', 'window-close-v2', 'sweep-into-v2', 'drawer-close-v2', 'plate-slide-side-v2', 'soccer-v2', 'reach-wall-v2', 'door-lock-v2', 'coffee-button-v2'],
             ['sweep-into-v2', 'plate-slide-side-v2', 'peg-unplug-side-v2', 'window-open-v2', 'door-lock-v2', 'reach-v2', 'reach-wall-v2', 'plate-slide-back-v2', 'door-close-v2', 'drawer-close-v2'],
             ['handle-pull-v2', 'soccer-v2', 'handle-pull-side-v2', 'button-press-v2', 'push-v2', 'window-close-v2', 'coffee-button-v2', 'hand-insert-v2', 'plate-slide-back-side-v2', 'coffee-push-v2'],
             ['window-open-v2', 'push-v2', 'reach-v2', 'plate-slide-back-v2', 'handle-pull-side-v2', 'soccer-v2', 'sweep-into-v2', 'coffee-button-v2', 'window-close-v2', 'coffee-push-v2'],
             ['drawer-close-v2', 'peg-unplug-side-v2', 'plate-slide-side-v2', 'door-lock-v2', 'plate-slide-back-side-v2', 'reach-wall-v2', 'handle-pull-v2', 'button-press-v2', 'hand-insert-v2', 'door-close-v2'],
             ['door-lock-v2', 'coffee-button-v2', 'hand-insert-v2', 'window-close-v2', 'button-press-v2', 'plate-slide-back-side-v2', 'plate-slide-back-v2', 'window-open-v2', 'reach-wall-v2', 'soccer-v2'],
             ['drawer-close-v2', 'handle-pull-v2', 'push-v2', 'plate-slide-side-v2', 'coffee-push-v2', 'handle-pull-side-v2', 'sweep-into-v2', 'reach-v2', 'peg-unplug-side-v2', 'door-close-v2'],
             ['hand-insert-v2', 'door-lock-v2', 'window-close-v2', 'reach-wall-v2', 'coffee-push-v2', 'drawer-close-v2', 'plate-slide-back-v2', 'handle-pull-v2', 'button-press-v2', 'sweep-into-v2'],
             ['reach-v2', 'push-v2', 'plate-slide-back-side-v2', 'plate-slide-side-v2', 'door-close-v2', 'soccer-v2', 'peg-unplug-side-v2', 'window-open-v2', 'handle-pull-side-v2', 'coffee-button-v2']]


RPO20_SEQ = [
['door-lock-v2', 'handle-press-v2', 'handle-press-side-v2', 'button-press-v2', 'door-close-v2', 'hand-insert-v2', 'reach-v2', 'plate-slide-v2', 'handle-pull-side-v2', 'plate-slide-back-side-v2', 'plate-slide-back-v2', 'soccer-v2', 'sweep-into-v2', 'reach-wall-v2', 'window-open-v2', 'coffee-button-v2', 'coffee-push-v2', 'peg-unplug-side-v2', 'window-close-v2', 'plate-slide-side-v2'],
['plate-slide-side-v2', 'plate-slide-back-v2', 'handle-press-side-v2', 'peg-unplug-side-v2', 'handle-pull-v2', 'reach-wall-v2', 'plate-slide-back-side-v2', 'button-press-v2', 'soccer-v2', 'hand-insert-v2', 'door-lock-v2', 'push-v2', 'window-close-v2', 'button-press-topdown-wall-v2', 'drawer-close-v2', 'sweep-into-v2', 'reach-v2', 'coffee-button-v2', 'coffee-push-v2', 'door-close-v2'],
['hand-insert-v2', 'peg-unplug-side-v2', 'handle-pull-side-v2', 'handle-press-v2', 'button-press-v2', 'coffee-push-v2', 'plate-slide-back-v2', 'handle-pull-v2', 'button-press-topdown-wall-v2', 'push-v2', 'plate-slide-v2', 'door-close-v2', 'reach-v2', 'window-open-v2', 'coffee-button-v2', 'window-close-v2', 'drawer-close-v2', 'soccer-v2', 'plate-slide-side-v2', 'plate-slide-back-side-v2'],
['handle-pull-side-v2', 'button-press-v2', 'window-open-v2', 'door-close-v2', 'reach-wall-v2', 'push-v2', 'hand-insert-v2', 'drawer-close-v2', 'handle-press-side-v2', 'handle-press-v2', 'door-lock-v2', 'plate-slide-back-side-v2', 'window-close-v2', 'sweep-into-v2', 'button-press-topdown-wall-v2', 'coffee-button-v2', 'soccer-v2', 'handle-pull-v2', 'plate-slide-back-v2', 'plate-slide-v2'],
['soccer-v2', 'coffee-button-v2', 'handle-press-v2', 'handle-pull-side-v2', 'plate-slide-back-v2', 'door-lock-v2', 'drawer-close-v2', 'reach-v2', 'peg-unplug-side-v2', 'plate-slide-v2', 'reach-wall-v2', 'handle-pull-v2', 'push-v2', 'plate-slide-side-v2', 'coffee-push-v2', 'button-press-topdown-wall-v2', 'hand-insert-v2', 'sweep-into-v2', 'window-open-v2', 'handle-press-side-v2'],
['door-close-v2', 'reach-wall-v2', 'coffee-push-v2', 'sweep-into-v2', 'door-lock-v2', 'plate-slide-v2', 'plate-slide-side-v2', 'peg-unplug-side-v2', 'handle-press-side-v2', 'handle-pull-v2', 'window-close-v2', 'push-v2', 'plate-slide-back-side-v2', 'reach-v2', 'handle-press-v2', 'window-open-v2', 'button-press-topdown-wall-v2', 'handle-pull-side-v2', 'drawer-close-v2', 'button-press-v2'],
['window-open-v2', 'window-close-v2', 'handle-pull-v2', 'push-v2', 'plate-slide-back-v2', 'button-press-v2', 'reach-v2', 'plate-slide-v2', 'coffee-button-v2', 'handle-pull-side-v2', 'hand-insert-v2', 'reach-wall-v2', 'drawer-close-v2', 'plate-slide-back-side-v2', 'sweep-into-v2', 'button-press-topdown-wall-v2', 'door-close-v2', 'plate-slide-side-v2', 'door-lock-v2', 'coffee-push-v2'],
['door-close-v2', 'soccer-v2', 'drawer-close-v2', 'handle-pull-side-v2', 'plate-slide-back-v2', 'hand-insert-v2', 'coffee-push-v2', 'reach-wall-v2', 'peg-unplug-side-v2', 'button-press-topdown-wall-v2', 'plate-slide-side-v2', 'reach-v2', 'window-close-v2', 'sweep-into-v2', 'button-press-v2', 'coffee-button-v2', 'handle-press-side-v2', 'handle-pull-v2', 'window-open-v2', 'handle-press-v2'],
['handle-pull-side-v2', 'push-v2', 'plate-slide-back-v2', 'plate-slide-side-v2', 'peg-unplug-side-v2', 'reach-wall-v2', 'sweep-into-v2', 'door-lock-v2', 'plate-slide-v2', 'window-open-v2', 'handle-press-v2', 'hand-insert-v2', 'handle-press-side-v2', 'handle-pull-v2', 'soccer-v2', 'drawer-close-v2', 'reach-v2', 'button-press-v2', 'window-close-v2', 'plate-slide-back-side-v2'],
['peg-unplug-side-v2', 'handle-press-side-v2', 'reach-wall-v2', 'door-close-v2', 'button-press-topdown-wall-v2', 'reach-v2', 'handle-pull-v2', 'drawer-close-v2', 'plate-slide-side-v2', 'coffee-button-v2', 'window-close-v2', 'handle-press-v2', 'door-lock-v2', 'coffee-push-v2', 'window-open-v2', 'plate-slide-back-side-v2', 'button-press-v2', 'push-v2', 'plate-slide-v2', 'soccer-v2']
]





GOOD_RPO_SEQS = [
    ['sweep-into-v2', 'door-close-v2', 'drawer-close-v2', 'button-press-v2', 'window-close-v2', 'hand-insert-v2', 'soccer-v2', 'handle-pull-side-v2'],
    ['peg-unplug-side-v2', 'soccer-v2', 'plate-slide-side-v2', 'plate-slide-back-side-v2', 'reach-wall-v2', 'reach-v2', 'drawer-close-v2', 'coffee-button-v2'],
    ['window-close-v2', 'window-open-v2', 'button-press-topdown-wall-v2', 'handle-press-v2', 'coffee-button-v2', 'handle-press-side-v2', 'push-v2', 'plate-slide-back-v2'],
    ['plate-slide-v2', 'door-close-v2', 'sweep-into-v2', 'door-lock-v2', 'coffee-push-v2', 'handle-press-v2', 'hand-insert-v2', 'reach-wall-v2'],
    ['coffee-button-v2', 'soccer-v2', 'plate-slide-v2', 'handle-pull-v2', 'window-open-v2', 'handle-press-v2', 'handle-pull-side-v2', 'drawer-close-v2'],
    ['handle-pull-side-v2', 'handle-press-v2', 'window-open-v2', 'handle-pull-v2', 'window-close-v2', 'coffee-button-v2', 'plate-slide-v2', 'coffee-push-v2'],
    ['button-press-v2', 'button-press-topdown-wall-v2', 'handle-press-side-v2', 'reach-wall-v2', 'hand-insert-v2', 'plate-slide-side-v2', 'peg-unplug-side-v2', 'push-v2'],
    ['peg-unplug-side-v2', 'reach-wall-v2', 'plate-slide-side-v2', 'door-close-v2', 'door-lock-v2', 'handle-press-side-v2', 'reach-v2', 'plate-slide-back-side-v2'],
    ['coffee-push-v2', 'handle-pull-v2', 'sweep-into-v2', 'plate-slide-back-v2', 'plate-slide-back-side-v2', 'reach-v2', 'plate-slide-v2', 'window-open-v2'],
    ['button-press-topdown-wall-v2', 'push-v2', 'drawer-close-v2', 'door-lock-v2', 'plate-slide-back-v2', 'door-close-v2', 'sweep-into-v2', 'button-press-v2']
    ]














if __name__ == "__main__":




    import random
    import numpy as np
    NUM_TASKS_PER_SEQ = 10
    NUM_SEQ = 20
    total_tasks = NUM_TASKS_PER_SEQ * NUM_SEQ




    all_seq_tasks = []


    candidate_set = set(GOOD_RPO_ENVS)


    for i_seq in range(NUM_SEQ):
        one_set_tasks = set([])
        num_tasks_to_sample = NUM_TASKS_PER_SEQ

        if len(candidate_set) < NUM_TASKS_PER_SEQ:
            one_set_tasks = one_set_tasks.union(candidate_set)
            num_tasks_to_sample -= len(candidate_set)
            candidate_set = set(GOOD_RPO_ENVS)


        sampled_set_tasks = random.sample(list(candidate_set.difference(one_set_tasks)), num_tasks_to_sample)
        one_set_tasks = one_set_tasks.union(sampled_set_tasks)


        candidate_set = candidate_set.difference(sampled_set_tasks)


        all_seq_tasks.append(one_set_tasks)



    all_seq_tasks = [list(task_set) for task_set in all_seq_tasks]
    for task_lst in all_seq_tasks:
        random.shuffle(task_lst)


    print(all_seq_tasks)




















































































    import collections
    seq_list = RPO10_SEQ

    for lst in seq_list:
        print(len(lst))

    flat_list = [
        x
        for xs in seq_list
        for x in xs
    ]



    counter = collections.Counter(flat_list)
    print(counter)