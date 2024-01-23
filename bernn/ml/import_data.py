import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from bernn.utils.utils import plot_confusion_matrix, scale_data, get_unique_labels, to_csv, scale_data_per_batch


def get_harvard(args, path='data'):
    """

    Args:
        path: Path where the csvs can be loaded. The folder designated by path needs to contain at least
                   one file named train_inputs.csv (when using --use_valid=0 and --use_test=0). When using
                   --use_valid=1 and --use_test=1, it must also contain valid_inputs.csv and test_inputs.csv.

    Returns:
        Nothing. The data is stored in self.data
    """
    not_zero_cols = []
    unique_labels = np.array([])
    data = {}
    for info in ['inputs', 'names', 'labels', 'cats', 'batches', 'meta', 'orders']:
        data[info] = {}
        for group in ['all', 'train', 'valid', 'test']:
            data[info][group] = np.array([])
    for group in ['train', 'test', 'valid']:
        if group == 'test':
            if args.groupkfold:
                skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=41)
                train_nums = np.arange(0, len(data['labels']['train']))
                train_inds, test_inds = skf.split(train_nums, data['labels']['train'],
                                                  data['batches']['train']).__next__()
            else:
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=41)
                train_nums = np.arange(0, len(data['labels']['train']))
                train_inds, test_inds = skf.split(train_nums, data['labels']['train']).__next__()
            data['inputs']['train'], data['inputs']['test'] = data['inputs']['train'].iloc[train_inds], \
                                                              data['inputs']['train'].iloc[test_inds]
            data['names']['train'], data['names']['test'] = data['names']['train'].iloc[train_inds], \
                                                            data['names']['train'].iloc[test_inds]
            data['labels']['train'], data['labels']['test'] = data['labels']['train'][train_inds], \
                                                              data['labels']['train'][test_inds]
            data['batches']['train'], data['batches']['test'] = data['batches']['train'][train_inds], \
                                                                data['batches']['train'][test_inds]
            data['meta']['train'], data['meta']['test'] = data['meta']['train'].iloc[train_inds], \
                                                                data['meta']['train'].iloc[test_inds]
            data['cats']['train'], data['cats']['test'] = \
                data['cats']['train'][train_inds], data['cats']['train'][test_inds]

        elif group == 'valid':
            if args.groupkfold:
                skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=41)
                train_nums = np.arange(0, len(data['labels']['train']))
                train_inds, test_inds = skf.split(train_nums, data['labels']['train'],
                                                  data['batches']['train']).__next__()
            else:
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=41)
                train_nums = np.arange(0, len(data['labels']['train']))
                train_inds, test_inds = skf.split(train_nums, data['labels']['train']).__next__()
            data['inputs']['train'], data['inputs']['valid'] = data['inputs']['train'].iloc[train_inds], \
                                                              data['inputs']['train'].iloc[test_inds]
            data['names']['train'], data['names']['valid'] = data['names']['train'].iloc[train_inds], \
                                                            data['names']['train'].iloc[test_inds]
            data['labels']['train'], data['labels']['valid'] = data['labels']['train'][train_inds], \
                                                              data['labels']['train'][test_inds]
            data['batches']['train'], data['batches']['valid'] = data['batches']['train'][train_inds], \
                                                                data['batches']['train'][test_inds]
            data['meta']['train'], data['meta']['valid'] = data['meta']['train'].iloc[train_inds], \
                                                                data['meta']['train'].iloc[test_inds]
            data['cats']['train'], data['cats']['valid'] = \
                data['cats']['train'][train_inds], data['cats']['train'][test_inds]

        else:
            meta = pd.read_csv(
                f"{path}/AllSamples_Oct2022/subjects_experiment_ATN_verified_diagnosis.csv", sep=","
            )
            meta_names = meta.loc[:, 'SampleID']
            meta_labels = meta.loc[:, 'ATN_diagnosis']
            meta_atn = meta.loc[:, 'CSF ATN Status Binary']
            meta_gender = meta.loc[:, 'Gender']
            meta_age = meta.loc[:, 'Age at time of LP (yrs)']
            meta_not_nans = [i for i, x in enumerate(meta_labels.isna()) if not x]
            meta_names, meta_labels = meta_names.iloc[meta_not_nans], meta_labels.iloc[meta_not_nans]
            meta_gender, meta_age, meta_atn = meta_gender.iloc[meta_not_nans], meta_age.iloc[meta_not_nans], \
                meta_atn.iloc[meta_not_nans]
            meta_nans = [i for i, x in enumerate(meta_atn.isna()) if x]
            meta_atn.iloc[meta_nans] = "A- T- N-"
            meta_gender = np.array([1 if x == 'Female' else 0 for i, x in enumerate(meta_gender)])
            matrix = pd.read_csv(
                f"{path}/AllSamples_Oct2022/DIANN/{args.csv_file}", sep=','
            )
            matrix.index = matrix['Unnamed: 0']
            matrix = matrix.iloc[:, 1:].fillna(0)
            if args.remove_zeros:
                mask1 = (matrix == 0).mean(axis=1) < 0.2
                matrix = matrix.loc[mask1]
            matrix.iloc[:] = np.log1p(matrix.values)
            names = [x.split("\\")[-1].split("_")[1] for x in matrix.columns]
            names = np.array(["_".join(x.split("-")) for x in names])
            batches = np.array([int(x.split("\\")[-1].split("-")[1].split("_")[0]) for x in matrix.columns])
            meta_names2 = np.array([name.split('_')[0] for name in meta_names.values])
            names2 = np.array([name.split('_')[0] for name in names])
            meta_pos = [
                np.argwhere(name == meta_names2)[0][0]
                for i, name in enumerate(names2) if name in meta_names2.tolist()
            ]
            pool_pos = [i for i, name in enumerate(names) if name.split("_")[0] == 'Pool']
            pool_meta_pos = [i for i, name in enumerate(meta_names) if name.split("_")[0] == 'Pool']
            pos = np.concatenate([
                np.argwhere(name == names2)
                for i, name in enumerate(meta_names2) if name in names2.tolist()
            ]).squeeze()

            data['inputs'][group] = matrix.iloc[:, pos].T
            data['meta'][group] = pd.DataFrame(meta_age.iloc[meta_pos].to_numpy(), columns=['Age'],
                                               index=data['inputs'][group].index)
            data['meta'][group] = pd.concat((data['meta'][group],
                                             pd.DataFrame(meta_gender[meta_pos], columns=['Gender'],
                                                          index=data['inputs'][group].index)), 1)
            data['names'][group] = meta_names.iloc[meta_pos]
            data['labels'][group] = meta_labels.iloc[meta_pos].to_numpy()
            data['batches'][group] = batches[pos]
            data['orders'][group] = np.array([x for x in range(len(data['batches'][group]))])

            data['inputs'][f"{group}_pool"] = matrix.iloc[:, pool_pos].T
            data['meta'][f"{group}_pool"] = pd.DataFrame(np.zeros(len(pool_pos)) - 1, columns=['Age'],
                                                         index=data['inputs'][f"{group}_pool"].index)
            data['meta'][f"{group}_pool"] = pd.concat((data['meta'][f"{group}_pool"],
                                                       pd.DataFrame(np.zeros(len(pool_pos)) + 0.5,
                                                                    columns=['Gender'],
                                                                    index=data['inputs'][f"{group}_pool"].index)),
                                                      1)
            data['names'][f"{group}_pool"] = np.array([f'pool_{i}' for i, _ in enumerate(pool_pos)])
            # MUST BE REPLACED WITH REAL ORDERS
            data['labels'][f"{group}_pool"] = np.array([f'pool' for _ in pool_pos])
            data['batches'][f"{group}_pool"] = batches[pool_pos]
            data['orders'][f"{group}_pool"] = np.array([x for x in range(len(data['batches'][f"{group}_pool"]))])
            data['cats'][f"{group}_pool"] = np.array(
                [len(np.unique(data['labels'][group])) for _ in batches[pool_pos]])
            contaminants = pd.read_csv('data/AllSamples_Oct2022/contaminants.csv').values.squeeze()
            features = data['inputs'][group].columns
            features_to_keep = [x for x in features if x not in contaminants]
            data['inputs'][group] = data['inputs'][group].loc[:, features_to_keep]
            data['inputs'][f"{group}_pool"] = data['inputs'][f"{group}_pool"].loc[:, features_to_keep]
            columns = data['inputs'][group].columns
            if args.strategy == 'As':
                data['cats'][group] = np.array(data['meta'][group]['As'])
                data['labels'][group] = np.array(['A+' if x == 1 else "A-" for x in data['meta'][group]['As']])
                unique_labels = np.array(get_unique_labels(data['labels'][group]).tolist())
                data['cats'][f"{group}_pool"] = np.array(
                    [len(unique_labels) for _ in data['meta'][f"{group}_pool"]['As']])
                data['labels'][f"{group}_pool"] = np.array(['pool' for _ in data['meta'][f"{group}_pool"]['As']])
                unique_labels = np.array(get_unique_labels(data['labels'][group]).tolist() + ['pool'])
            elif args.strategy == 'Ts':
                data['cats'][group] = np.array(
                    [1 if x.split(' ')[1] == 'T+' else 0 for i, x in enumerate(data['labels'][group])])
                data['labels'][group] = np.array([x.split(' ')[1] for i, x in enumerate(data['labels'][group])])
            elif args.strategy == 'Ns':
                data['cats'][group] = np.array(
                    [1 if x.split(' ')[2] == 'N+' else 0 for i, x in enumerate(data['labels'][group])])
                data['labels'][group] = np.array([x.split(' ')[2] for i, x in enumerate(data['labels'][group])])
            elif args.strategy == 'dementia':
                data['labels'][group] = np.array(
                    ['DEM/MCI' if 'DEM' in x or 'MCI' in x else x for i, x in enumerate(data['labels'][group])])
                unique_labels = get_unique_labels(data['labels'][group])
                data['cats'][group] = np.array(
                    [np.where(x == unique_labels)[0][0] for i, x in enumerate(data['labels'][group])])
            elif args.strategy == 'none':
                data['labels'][group] = np.array([x.split('-')[0] for i, x in enumerate(data['labels'][group])])
                unique_labels = get_unique_labels(data['labels'][group])
                data['cats'][group] = np.array(
                    [np.where(x == unique_labels)[0][0] for i, x in enumerate(data['labels'][group])])
            elif args.strategy in ['no', 'ova']:
                unique_labels = get_unique_labels(data['labels'][group])
                unique_labels = np.array(unique_labels.tolist() + ['pool'])
                data['cats'][group] = np.array(
                    [np.where(x == unique_labels)[0][0] for i, x in enumerate(data['labels'][group])])
                data['cats'][f"{group}_pool"] = np.array(
                    [np.where(x == unique_labels)[0][0] for i, x in enumerate(data['labels'][f"{group}_pool"])])
            elif args.strategy == '3':
                data['labels'][group] = np.array(
                    ['CU' if 'NPH' in x else x.split('-')[0] for i, x in enumerate(data['labels'][group])])
                unique_labels = np.array(get_unique_labels(data['labels'][group]).tolist() + ['pool'])
                data['cats'][group] = np.array(
                    [np.where(x == unique_labels)[0][0] if x in unique_labels else len(unique_labels) for i, x in
                     enumerate(data['labels'][group])])
                data['cats'][f"{group}_pool"] = np.array(
                    [np.where(x == unique_labels)[0][0] for i, x in enumerate(data['labels'][f"{group}_pool"])])

            elif args.strategy == '2':
                data['labels'][group] = np.array(
                    ['CU' if 'NPH' in x else x.split('-')[0] for i, x in enumerate(data['labels'][group])])
                unique_labels = np.array(get_unique_labels(data['labels'][group]).tolist()).tolist()
                _ = unique_labels.pop(np.argwhere(np.array(unique_labels) == 'MCI').squeeze())
                unique_labels = np.array(unique_labels + ['pool', 'MCI'])
                data['cats'][group] = np.array(
                    [np.where(x == unique_labels)[0][0] if x in unique_labels else len(unique_labels) + 1 for i, x
                     in enumerate(data['labels'][group])])
                data['cats'][f"{group}_pool"] = np.array(
                    [np.where(x == unique_labels)[0][0] for i, x in enumerate(data['labels'][f"{group}_pool"])])

            elif args.strategy == 'NPH_CU_DEM':
                data['labels'][group] = np.array(
                    [x.split('-')[0] for i, x in enumerate(data['labels'][group])])
                unique_labels = np.array(get_unique_labels(data['labels'][group]).tolist()).tolist()
                _ = unique_labels.pop(np.argwhere(np.array(unique_labels) == 'MCI').squeeze())
                unique_labels = np.array(unique_labels + ['pool', 'MCI'])
                data['cats'][group] = np.array(
                    [np.where(x == unique_labels)[0][0] if x in unique_labels else len(unique_labels) + 1 for i, x
                     in enumerate(data['labels'][group])])
                data['cats'][f"{group}_pool"] = np.array(
                    [np.where(x == unique_labels)[0][0] for i, x in enumerate(data['labels'][f"{group}_pool"])])

            elif args.strategy == 'CU_DEM':
                data['labels'][group] = np.array(
                    [x.split('-')[0] for i, x in enumerate(data['labels'][group])])
                unique_labels = np.array(get_unique_labels(data['labels'][group]).tolist()).tolist()
                _ = unique_labels.pop(np.argwhere(np.array(unique_labels) == 'MCI').squeeze())
                _ = unique_labels.pop(np.argwhere(np.array(unique_labels) == 'NPH').squeeze())
                unique_labels = np.array(unique_labels + ['pool', 'NPH', 'MCI'])
                data['cats'][group] = np.array(
                    [np.where(x == unique_labels)[0][0] if x in unique_labels else len(unique_labels) + 1 for i, x
                     in enumerate(data['labels'][group])])
                data['cats'][f"{group}_pool"] = np.array(
                    [np.where(x == unique_labels)[0][0] for i, x in enumerate(data['labels'][f"{group}_pool"])])

            elif args.strategy == 'CU_DEM-AD':
                inds = [i for i, x in enumerate(data['labels'][group]) if x in ['CU', 'DEM-AD']]
                data['labels'][group] = data['labels'][group][inds]
                data['names'][group] = data['names'][group].iloc[inds]
                data['batches'][group] = data['batches'][group][inds]
                data['inputs'][group] = data['inputs'][group].iloc[inds]
                data['meta'][group] = data['meta'][group].iloc[inds]
                unique_labels = np.array(get_unique_labels(data['labels'][group]).tolist()).tolist()
                # unique_labels = np.array(unique_labels + ['pool', 'NPH', 'MCI'])
                unique_labels = np.array(unique_labels + ['pool'])
                data['cats'][group] = np.array(
                    [np.where(x == unique_labels)[0][0] if x in unique_labels else len(unique_labels) for i, x in
                     enumerate(data['labels'][group])])
                data['cats'][f"{group}_pool"] = np.array(
                    [np.where(x == unique_labels)[0][0] for i, x in enumerate(data['labels'][f"{group}_pool"])])

            if args.bad_batches != '':
                bad_batches = [int(x) for x in args.bad_batches.split(';')]
                good_batches = [x for x in np.unique(data['batches'][group]) if x not in bad_batches]
                inds_to_keep = np.concatenate(
                    [np.argwhere(data['batches'][group] == bad).squeeze() for bad in good_batches])
                for key in list(data.keys()):
                    if key in ['inputs', 'meta', 'names']:
                        data[key][group] = data[key][group].iloc[inds_to_keep]
                    else:
                        data[key][group] = data[key][group][inds_to_keep]

            if group != 'train' and len(not_zero_cols) > 0:
                data['inputs'][group] = data['inputs'][group].loc[:, not_zero_cols]
            else:
                if args.threshold == 1:
                    data['inputs'][group], not_zero_cols = keep_only_not_zeros(data['inputs'][group])
                elif args.threshold > 0:
                    data['inputs'][group], not_zero_cols = keep_not_zeros(data['inputs'][group],
                                                                          threshold=self.args.threshold)

    for key in list(data.keys()):
        if key in ['inputs', 'meta']:
            data[key]['all'] = pd.concat((
                data[key]['train'], data[key]['valid'], data[key]['test']
            ), 0)
        else:
            data[key]['all'] = np.concatenate((
                data[key]['train'], data[key]['valid'], data[key]['test']
            ), 0)

    unique_batches = np.unique(data['batches']['all'])
    for group in ['train', 'test', 'valid', 'all']:
        data['batches'][group] = np.array([np.argwhere(unique_batches == x)[0][0] for x in data['batches'][group]])

    return data, unique_labels, unique_batches


def get_prostate(args, path, seed=42):
    """

    Args:
        path: Path where the csvs can be loaded. The folder designated by path needs to contain at least
                   one file named train_inputs.csv (when using --use_valid=0 and --use_test=0). When using
                   --use_valid=1 and --use_test=1, it must also contain valid_inputs.csv and test_inputs.csv.

    Returns:
        Nothing. The data is stored in self.data
    """
    not_zero_cols = []
    unique_labels = np.array([])
    data = {}
    for info in ['inputs', 'meta', 'names', 'labels', 'cats', 'batches', 'orders']:
        data[info] = {}
        for group in ['all', 'train', 'test', 'valid']:
            data[info][group] = np.array([])
    for group in ['train', 'test', 'valid']:
        if group == 'test':
            if args.groupkfold:
                skf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)
                train_nums = np.arange(0, len(data['labels']['train']))
                train_inds, test_inds = skf.split(train_nums, data['labels']['train'],
                                                  data['batches']['train']).__next__()
            else:
                skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
                train_nums = np.arange(0, len(data['labels']['train']))
                train_inds, test_inds = skf.split(train_nums, data['labels']['train']).__next__()
            data['inputs']['train'], data['inputs']['test'] = data['inputs']['train'].iloc[train_inds], \
                data['inputs']['train'].iloc[test_inds]
            data['meta']['train'], data['meta']['test'] = data['meta']['train'].iloc[train_inds], \
                data['meta']['train'].iloc[test_inds]
            data['names']['train'], data['names']['test'] = data['names']['train'].iloc[train_inds], \
                data['names']['train'].iloc[test_inds]
            data['labels']['train'], data['labels']['test'] = data['labels']['train'][train_inds], \
                data['labels']['train'][test_inds]
            data['batches']['train'], data['batches']['test'] = data['batches']['train'][train_inds], \
                data['batches']['train'][test_inds]
            data['orders']['train'], data['orders']['test'] = data['orders']['train'][train_inds], \
                data['orders']['train'][test_inds]
            data['cats']['train'], data['cats']['test'] = \
                data['cats']['train'][train_inds], data['cats']['train'][test_inds]

        elif group == 'valid':
            if args.groupkfold:
                skf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)
                train_nums = np.arange(0, len(data['labels']['train']))
                train_inds, test_inds = skf.split(train_nums, data['labels']['train'],
                                                  data['batches']['train']).__next__()
            else:
                skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
                train_nums = np.arange(0, len(data['labels']['train']))
                train_inds, test_inds = skf.split(train_nums, data['labels']['train']).__next__()
            data['inputs']['train'], data['inputs']['valid'] = data['inputs']['train'].iloc[train_inds], \
                data['inputs']['train'].iloc[test_inds]
            data['meta']['train'], data['meta']['valid'] = data['meta']['train'].iloc[train_inds], \
                data['meta']['train'].iloc[test_inds]
            data['names']['train'], data['names']['valid'] = data['names']['train'].iloc[train_inds], \
                data['names']['train'].iloc[test_inds]
            data['labels']['train'], data['labels']['valid'] = data['labels']['train'][train_inds], \
                data['labels']['train'][test_inds]
            data['batches']['train'], data['batches']['valid'] = data['batches']['train'][train_inds], \
                data['batches']['train'][test_inds]
            data['orders']['train'], data['orders']['valid'] = data['orders']['train'][train_inds], \
                data['orders']['train'][test_inds]
            data['cats']['train'], data['cats']['valid'] = \
                data['cats']['train'][train_inds], data['cats']['train'][test_inds]

        else:
            matrix = pd.read_csv(
                f"{path}/training_data_three_Pca_cohorts.csv", sep="\t", index_col=0
            )
            names = pd.DataFrame(matrix.index).loc[:, 'patient']
            batches = matrix.loc[:, 'Set']
            unique_batches = batches.unique()
            batches = np.stack([np.argwhere(x == unique_batches).squeeze() for x in batches])
            labels = matrix.loc[:, 'BCR_60']
            orders = np.array([0 for _ in batches])
            matrix = matrix.iloc[:, 1:-4].fillna(0)
            if args.remove_zeros:
                mask1 = (matrix == 0).mean(axis=0) < 0.2
                matrix = matrix.loc[:, mask1]
            matrix.iloc[:] = np.log1p(matrix.values)
            # pool_pos = [i for i, name in enumerate(names.values.flatten()) if 'QC' in name]
            pos = [i for i, name in enumerate(names.values.flatten()) if 'QC' not in name]
            data['inputs'][group] = matrix.iloc[pos]
            data['names'][group] = names
            data['labels'][group] = labels.to_numpy()[pos]
            data['batches'][group] = batches[pos]
            # This is juste to make the pipeline work. Meta should be 0 for the amide dataset
            data['meta'][group] = data['inputs'][group].iloc[:, :2]
            data['orders'][group] = orders[pos]
            unique_labels = get_unique_labels(data['labels'][group])
            data['cats'][group] = data['labels'][group]

    for key in list(data.keys()):
        if key in ['inputs', 'meta']:
            data[key]['all'] = pd.concat((
                data[key]['train'], data[key]['valid'], data[key]['test']
            ), 0)
        else:
            data[key]['all'] = np.concatenate((
                data[key]['train'], data[key]['valid'], data[key]['test']
            ), 0)

    unique_batches = np.unique(data['batches']['all'])
    for group in ['train', 'test', 'valid', 'all']:
        data['batches'][group] = np.array([np.argwhere(unique_batches == x)[0][0] for x in data['batches'][group]])

    unique_labels = unique_labels.tolist()
    unique_batches = unique_batches

    return data, unique_labels, unique_batches


