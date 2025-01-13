def get_subject_clusters_assignment(cluster_labels, points_subjects_ids, subjects):
    '''
    Return a dictionary with cluster labels as ids and list of subjects ids as values
    Assign cluster to subject with majority voting
    '''
    assert len(cluster_labels) == len(points_subjects_ids)
    subjects_clusters_dict = {}
    for i, subj in enumerate(points_subjects_ids):
        if subj not in subjects_clusters_dict:
            subjects_clusters_dict[subj] = [cluster_labels[i]]
        else:
            subjects_clusters_dict[subj].append(cluster_labels[i])

    cluster_subjects_dict = {}
    for subj in subjects:
        most_frequent_cluster = max(subjects_clusters_dict[subj],key=subjects_clusters_dict[subj].count)
        if most_frequent_cluster not in cluster_subjects_dict:
            cluster_subjects_dict[most_frequent_cluster] = [subj]
        else:
            cluster_subjects_dict[most_frequent_cluster].append(subj)
    sum_clusters_subjects = sum([len(cluster_subjects_dict[key]) for key in cluster_subjects_dict])
    assert sum_clusters_subjects == len(subjects)
    return cluster_subjects_dict