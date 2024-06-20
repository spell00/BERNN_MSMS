import numpy as np
import pandas as pd
from bernn.utils.utils import get_unique_labels
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from bernn.dl.models.pytorch.utils.dataset import MSCSV, MS2CSV

def get_alzheimer(path, args, seed=42):
    """
    Args:
        path: Path where data is located.
        args: arguments from the command line to be used in the data getter

    Returns:
        data, unique_labels, unique_batches
    """
    data = {}
    unique_labels = np.array([])
    for info in ['inputs', 'meta', 'names', 'labels', 'cats', 'batches', 'orders', 'sets']:
        data[info] = {}
        for group in ['all', 'train', 'test', 'valid']:
            data[info][group] = np.array([])
    for group in ['train', 'valid']:
        if group == 'valid':
            if args.groupkfold:
                skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
                train_nums = np.arange(0, len(data['labels']['train']))
                # Remove samples from unwanted batches
                splitter = skf.split(train_nums, data['labels']['train'], data['batches']['train'])
                skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
                train_nums_pool = np.arange(0, len(data['labels']['train_pool']))
                pool_splitter = skf.split(train_nums_pool, data['labels']['train_pool'],
                                                   data['batches']['train_pool'])

            else:
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
                train_nums = np.arange(0, len(data['labels']['train']))
                splitter = skf.split(train_nums, data['labels']['train']).__next__()
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
                train_nums_pool = np.arange(0, len(data['labels']['train_pool']))
                pool_splitter = skf.split(train_nums_pool, data['labels']['train_pool'])

            _, valid_inds = splitter.__next__()
            _, test_inds = splitter.__next__()
            train_inds = [x for x in train_nums if x not in np.concatenate((valid_inds, test_inds))]

            data['inputs']['train'], data['inputs']['valid'], data['inputs']['test'] = data['inputs']['train'].iloc[train_inds], \
                data['inputs']['train'].iloc[valid_inds], data['inputs']['train'].iloc[test_inds]
            data['meta']['train'], data['meta']['valid'], data['meta']['test'] = data['meta']['train'].iloc[train_inds], \
                data['meta']['train'].iloc[valid_inds], data['meta']['train'].iloc[test_inds]
            data['labels']['train'], data['labels']['valid'], data['labels']['test'] = data['labels']['train'][train_inds], \
                data['labels']['train'][valid_inds], data['labels']['train'][test_inds]
            data['names']['train'], data['names']['valid'], data['names']['test'] = data['names']['train'].iloc[train_inds], \
                data['names']['train'].iloc[valid_inds], data['names']['train'].iloc[test_inds]
            data['orders']['train'], data['orders']['valid'], data['orders']['test'] = data['orders']['train'][train_inds], \
                data['orders']['train'][valid_inds], data['orders']['train'][test_inds]
            data['batches']['train'], data['batches']['valid'], data['batches']['test'] = data['batches']['train'][train_inds], \
                data['batches']['train'][valid_inds], data['batches']['train'][test_inds]
            data['cats']['train'], data['cats']['valid'], data['cats']['test'] = data['cats']['train'][train_inds], data['cats']['train'][
                valid_inds], data['cats']['train'][test_inds]

            _, valid_inds = pool_splitter.__next__()
            _, test_inds = pool_splitter.__next__()
            train_inds = [x for x in train_nums_pool if x not in np.concatenate((valid_inds, test_inds))]
            data['inputs']['train_pool'], data['inputs']['valid_pool'], data['inputs']['test_pool'], = data['inputs']['train_pool'].iloc[train_inds], \
                data['inputs']['train_pool'].iloc[valid_inds], data['inputs']['train_pool'].iloc[test_inds]
            data['meta']['train_pool'], data['meta']['valid_pool'], data['meta']['test_pool'], = data['meta']['train_pool'].iloc[train_inds], \
                data['meta']['train_pool'].iloc[valid_inds], data['meta']['train_pool'].iloc[test_inds]
            data['labels']['train_pool'], data['labels']['valid_pool'], data['labels']['test_pool'], = data['labels']['train_pool'][train_inds], \
                data['labels']['train_pool'][valid_inds], data['labels']['train_pool'][test_inds]
            data['names']['train_pool'], data['names']['valid_pool'], data['names']['test_pool'], = data['names']['train_pool'][train_inds], \
                data['names']['train_pool'][valid_inds], data['names']['train_pool'][test_inds]
            data['orders']['train_pool'], data['orders']['valid_pool'], data['orders']['test_pool'], = data['orders']['train_pool'][train_inds], \
                data['orders']['train_pool'][valid_inds], data['orders']['train_pool'][test_inds]
            data['batches']['train_pool'], data['batches']['valid_pool'], data['batches']['test_pool'], = data['batches']['train_pool'][train_inds], \
                data['batches']['train_pool'][valid_inds], data['batches']['train_pool'][test_inds]
            data['cats']['train_pool'], data['cats']['valid_pool'], data['cats']['test_pool'], = data['cats']['train_pool'][train_inds], data['cats']['train_pool'][
                valid_inds], data['cats']['train_pool'][test_inds]

        else:
            meta = pd.read_csv(
                f"{path}/subjects_experiment_ATN_verified_diagnosis.csv", sep=","
            )
            meta_names = pd.Series([x.split('_')[1].split('-')[0] for x in meta.loc[:, 'sample_id']])
            meta_labels = meta.loc[:, 'ATN_diagnosis']
            # meta_atn = meta.loc[:, 'CSF ATN Status Binary']
            meta_gender = meta.loc[:, 'Gender']
            meta_age = meta.loc[:, 'Age at time of LP (yrs)']
            meta_not_nans = [i for i, x in enumerate(meta_labels.isna()) if not x]
            meta_names, meta_labels = meta_names.iloc[meta_not_nans], meta_labels.iloc[meta_not_nans]
            meta_gender, meta_age = meta_gender.iloc[meta_not_nans], meta_age.iloc[meta_not_nans]
            meta_gender = np.array([1 if x == 'Female' else 0 for i, x in enumerate(meta_gender)])
            matrix = pd.read_csv(
                f"{path}/{args.csv_file}", sep=','
            )
            matrix.index = matrix['Gene_id']
            matrix = matrix.iloc[:, 1:].fillna(0)
            if args.remove_zeros:
                mask1 = (matrix == 0).mean(axis=1) < 0.2
                matrix = matrix.loc[mask1]
            if args.log1p:
                matrix.iloc[:] = np.log1p(matrix.values)

            def impute_zero(peak):
                zero_mask = peak == 0
                if zero_mask.any():
                    new_x = peak.copy()
                    impute_value = peak.loc[~zero_mask].min()
                    new_x[zero_mask] = impute_value / 2
                    return new_x
                return peak

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
                for i, name in enumerate(np.unique(meta_names2)) if name in names2.tolist()
            ]).squeeze()

            data['inputs'][group] = matrix.iloc[:, pos].T
            if not args.zinb:
                data['inputs'][group] = data['inputs'][group].apply(impute_zero, axis=0)
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
                                                                    index=data['inputs'][f"{group}_pool"].index)), 1)
            data['names'][f"{group}_pool"] = np.array([f'pool_{i}' for i, _ in enumerate(pool_pos)])
            # MUST BE REPLACED WITH REAL ORDERS
            data['labels'][f"{group}_pool"] = np.array([f'pool' for _ in pool_pos])
            data['batches'][f"{group}_pool"] = batches[pool_pos]
            data['orders'][f"{group}_pool"] = np.array([x for x in range(len(data['batches'][f"{group}_pool"]))])
            data['cats'][f"{group}_pool"] = np.array(
                [len(np.unique(data['labels'][group])) for _ in batches[pool_pos]])
            contaminants = pd.read_csv(f'{path}/contaminants.csv').values.squeeze()
            features = data['inputs'][group].columns
            features_to_keep = [x for x in features if x not in contaminants]
            data['inputs'][group] = data['inputs'][group].loc[:, features_to_keep]
            data['inputs'][f"{group}_pool"] = data['inputs'][f"{group}_pool"].loc[:, features_to_keep]

            data['labels'][group] = np.array(
                [x for i, x in enumerate(data['labels'][group])])
            unique_labels = np.array(get_unique_labels(data['labels'][group]).tolist()).tolist()
            _ = unique_labels.pop(np.argwhere(np.array(unique_labels) == 'MCI-AD').squeeze())
            _ = unique_labels.pop(np.argwhere(np.array(unique_labels) == 'DEM-other').squeeze())
            _ = unique_labels.pop(np.argwhere(np.array(unique_labels) == 'MCI-other').squeeze())
            _ = unique_labels.pop(np.argwhere(np.array(unique_labels) == 'NPH').squeeze())
            # unique_labels = np.array(unique_labels + ['pool', 'NPH', 'MCI'])
            unique_labels = np.array(unique_labels + ['MCI-AD', 'MCI-other', 'NPH', 'DEM-other', 'pool'])
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

    for key in list(data['names'].keys()):
        data['sets'][key] = np.array([key for _ in data['names'][key]])
    for key in list(data.keys()):
        if key in ['inputs', 'meta']:
            data[key]['all'] = pd.concat((
                data[key]['train'], data[key]['valid'], data[key]['test'],
                data[key]['train_pool'], data[key]['valid_pool'], data[key]['test_pool'],
            ), 0)
            data[key]['all_pool'] = pd.concat((
                data[key]['train_pool'], data[key]['valid_pool'], data[key]['test_pool'],
            ), 0)
        else:
            data[key]['all'] = np.concatenate((
                data[key]['train'], data[key]['valid'], data[key]['test'],
                data[key]['train_pool'], data[key]['valid_pool'], data[key]['test_pool'],
            ), 0)
            data[key]['all_pool'] = np.concatenate((
                data[key]['train_pool'], data[key]['valid_pool'], data[key]['test_pool'],
            ), 0)

    unique_batches = np.unique(data['batches']['all'])
    for group in ['train', 'valid', 'test', 'train_pool', 'valid_pool', 'test_pool', 'all', 'all_pool']:
        data['batches'][group] = np.array([np.argwhere(unique_batches == x)[0][0] for x in data['batches'][group]])

    return data, unique_labels, unique_batches


def get_amide(path, args, seed=42):
    """
    Args:
        path: Path where data is located.
        args: arguments from the command line to be used in the data getter

    Returns:
        data, unique_labels, unique_batches
    """
    data = {}
    unique_labels = np.array([])
    for info in ['inputs', 'meta', 'names', 'labels', 'cats', 'batches', 'orders', 'sets']:
        data[info] = {}
        for group in ['all', 'train', 'test', 'valid']:
            data[info][group] = np.array([])
    for group in ['train', 'valid']:
        if group == 'valid':
            if args.groupkfold:
                skf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)
                train_nums = np.arange(0, len(data['labels']['train']))
                # Remove samples from unwanted batches
                splitter = skf.split(train_nums, data['labels']['train'], data['batches']['train'])
                skf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)
                train_nums_pool = np.arange(0, len(data['labels']['train_pool']))
                pool_splitter = skf.split(train_nums_pool, data['labels']['train_pool'],
                                                   data['batches']['train_pool'])

            else:
                skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
                train_nums = np.arange(0, len(data['labels']['train']))
                splitter = skf.split(train_nums, data['labels']['train']).__next__()
                skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
                train_nums_pool = np.arange(0, len(data['labels']['train_pool']))
                pool_splitter = skf.split(train_nums_pool, data['labels']['train_pool'])

            _, valid_inds = splitter.__next__()
            _, test_inds = splitter.__next__()
            train_inds = [x for x in train_nums if x not in np.concatenate((valid_inds, test_inds))]

            data['inputs']['train'], data['inputs']['valid'], data['inputs']['test'] = data['inputs']['train'].iloc[train_inds], \
                data['inputs']['train'].iloc[valid_inds], data['inputs']['train'].iloc[test_inds]
            data['meta']['train'], data['meta']['valid'], data['meta']['test'] = data['meta']['train'].iloc[train_inds], \
                data['meta']['train'].iloc[valid_inds], data['meta']['train'].iloc[test_inds]
            data['labels']['train'], data['labels']['valid'], data['labels']['test'] = data['labels']['train'][train_inds], \
                data['labels']['train'][valid_inds], data['labels']['train'][test_inds]
            data['names']['train'], data['names']['valid'], data['names']['test'] = data['names']['train'].iloc[train_inds], \
                data['names']['train'].iloc[valid_inds], data['names']['train'].iloc[test_inds]
            data['orders']['train'], data['orders']['valid'], data['orders']['test'] = data['orders']['train'][train_inds], \
                data['orders']['train'][valid_inds], data['orders']['train'][test_inds]
            data['batches']['train'], data['batches']['valid'], data['batches']['test'] = data['batches']['train'][train_inds], \
                data['batches']['train'][valid_inds], data['batches']['train'][test_inds]
            data['cats']['train'], data['cats']['valid'], data['cats']['test'] = data['cats']['train'][train_inds], data['cats']['train'][
                valid_inds], data['cats']['train'][test_inds]

            _, valid_inds = pool_splitter.__next__()
            _, test_inds = pool_splitter.__next__()
            train_inds = [x for x in train_nums_pool if x not in np.concatenate((valid_inds, test_inds))]
            data['inputs']['train_pool'], data['inputs']['valid_pool'], data['inputs']['test_pool'], = data['inputs']['train_pool'].iloc[train_inds], \
                data['inputs']['train_pool'].iloc[valid_inds], data['inputs']['train_pool'].iloc[test_inds]
            data['meta']['train_pool'], data['meta']['valid_pool'], data['meta']['test_pool'], = data['meta']['train_pool'].iloc[train_inds], \
                data['meta']['train_pool'].iloc[valid_inds], data['meta']['train_pool'].iloc[test_inds]
            data['labels']['train_pool'], data['labels']['valid_pool'], data['labels']['test_pool'], = data['labels']['train_pool'][train_inds], \
                data['labels']['train_pool'][valid_inds], data['labels']['train_pool'][test_inds]
            data['names']['train_pool'], data['names']['valid_pool'], data['names']['test_pool'], = data['names']['train_pool'][train_inds], \
                data['names']['train_pool'][valid_inds], data['names']['train_pool'][test_inds]
            data['orders']['train_pool'], data['orders']['valid_pool'], data['orders']['test_pool'], = data['orders']['train_pool'][train_inds], \
                data['orders']['train_pool'][valid_inds], data['orders']['train_pool'][test_inds]
            data['batches']['train_pool'], data['batches']['valid_pool'], data['batches']['test_pool'], = data['batches']['train_pool'][train_inds], \
                data['batches']['train_pool'][valid_inds], data['batches']['train_pool'][test_inds]
            data['cats']['train_pool'], data['cats']['valid_pool'], data['cats']['test_pool'], = data['cats']['train_pool'][train_inds], data['cats']['train_pool'][
                valid_inds], data['cats']['train_pool'][test_inds]

        else:
            matrix = pd.read_csv(
                f"{path}/amide_data.csv", sep=",", index_col=0
            )
            names = pd.DataFrame(matrix.index).loc[:, 0]
            batches = matrix.loc[:, 'batch']
            labels = matrix.loc[:, 'group']
            orders = matrix.loc[:, 'Injection_order']
            matrix = matrix.iloc[:, 3:].fillna(0)
            if args.remove_zeros:
                mask1 = (matrix == 0).mean(axis=0) < 0.2
                matrix = matrix.loc[:, mask1]
            if args.log1p:
                matrix.iloc[:] = np.log1p(matrix.values)

            def impute_zero(peak):
                zero_mask = peak == 0
                if zero_mask.any():
                    new_x = peak.copy()
                    impute_value = peak.loc[~zero_mask].min()
                    new_x[zero_mask] = impute_value / 2
                    return new_x
                return peak

            if not args.zinb:
                print('Imputing zeros.')
                matrix = matrix.apply(impute_zero, axis=0)
            pool_pos = [i for i, name in enumerate(names.values.flatten()) if 'QC' in name]
            pos = [i for i, name in enumerate(names.values.flatten()) if 'QC' not in name]
            data['inputs'][group] = matrix.iloc[pos]
            data['names'][group] = names
            data['labels'][group] = labels.to_numpy()[pos]
            data['batches'][group] = batches[pos]
            # This is juste to make the pipeline work. Meta should be 0 for the amide dataset
            data['meta'][group] = data['inputs'][group].iloc[:, :2]
            data['orders'][group] = orders[pos]

            data['inputs'][f"{group}_pool"] = matrix.iloc[pool_pos]
            data['names'][f"{group}_pool"] = np.array([f'pool_{i}' for i, _ in enumerate(pool_pos)])
            data['labels'][f"{group}_pool"] = np.array([f'pool' for _ in pool_pos])
            data['batches'][f"{group}_pool"] = batches[pool_pos]

            # This is juste to make the pipeline work. Meta should be 0 for the amide dataset
            data['meta'][f"{group}_pool"] = data['inputs'][f"{group}_pool"].iloc[:, :2]
            data['orders'][f"{group}_pool"] = orders[pool_pos]
            data['cats'][f"{group}_pool"] = np.array(
                [len(np.unique(data['labels'][group])) for _ in batches[pool_pos]])

            data['labels'][group] = np.array([x.split('-')[0] for i, x in enumerate(data['labels'][group])])
            unique_labels = np.concatenate((get_unique_labels(data['labels'][group]), np.array(['pool'])))
            data['cats'][group] = np.array(
                [np.where(x == unique_labels)[0][0] for i, x in enumerate(data['labels'][group])])
    for key in list(data['names'].keys()):
        data['sets'][key] = np.array([key for _ in data['names'][key]])
    for key in list(data.keys()):
        if key in ['inputs', 'meta']:
            data[key]['all'] = pd.concat((
                data[key]['train'], data[key]['valid'], data[key]['test'],
                data[key]['train_pool'], data[key]['valid_pool'], data[key]['test_pool'],
            ), 0)
            data[key]['all_pool'] = pd.concat((
                data[key]['train_pool'], data[key]['valid_pool'], data[key]['test_pool'],
            ), 0)
        else:
            data[key]['all'] = np.concatenate((
                data[key]['train'], data[key]['valid'], data[key]['test'],
                data[key]['train_pool'], data[key]['valid_pool'], data[key]['test_pool'],
            ), 0)
            data[key]['all_pool'] = np.concatenate((
                data[key]['train_pool'], data[key]['valid_pool'], data[key]['test_pool'],
            ), 0)

    unique_batches = np.unique(data['batches']['all'])
    for group in ['train', 'valid', 'test', 'train_pool', 'valid_pool', 'test_pool', 'all', 'all_pool']:
        data['batches'][group] = np.array([np.argwhere(unique_batches == x)[0][0] for x in data['batches'][group]])

    return data, unique_labels, unique_batches


def get_mice(path, args, seed=42):
    """
    Args:
        path: Path where data is located.
        args: arguments from the command line to be used in the data getter

    Returns:
        data, unique_labels, unique_batches
    """
    data = {}
    unique_labels = np.array([])
    for info in ['inputs', 'meta', 'names', 'labels', 'cats', 'batches', 'orders', 'sets']:
        data[info] = {}
        for group in ['all', 'train', 'test', 'valid']:
            data[info][group] = np.array([])
    for group in ['train', 'valid']:
        if group == 'valid':
            if args.groupkfold:
                skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
                train_nums = np.arange(0, len(data['labels']['train']))
                # Remove samples from unwanted batches
                splitter = skf.split(train_nums, data['labels']['train'], data['batches']['train'])
            else:
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
                train_nums = np.arange(0, len(data['labels']['train']))
                splitter = skf.split(train_nums, data['labels']['train']).__next__()

            _, valid_inds = splitter.__next__()
            _, test_inds = splitter.__next__()
            train_inds = [x for x in train_nums if x not in np.concatenate((valid_inds, test_inds))]

            data['inputs']['train'], data['inputs']['valid'], data['inputs']['test'] = data['inputs']['train'].iloc[train_inds], \
                data['inputs']['train'].iloc[valid_inds], data['inputs']['train'].iloc[test_inds]
            data['meta']['train'], data['meta']['valid'], data['meta']['test'] = data['meta']['train'].iloc[train_inds], \
                data['meta']['train'].iloc[valid_inds], data['meta']['train'].iloc[test_inds]
            data['labels']['train'], data['labels']['valid'], data['labels']['test'] = data['labels']['train'][train_inds], \
                data['labels']['train'][valid_inds], data['labels']['train'][test_inds]
            data['names']['train'], data['names']['valid'], data['names']['test'] = data['names']['train'].iloc[train_inds], \
                data['names']['train'].iloc[valid_inds], data['names']['train'].iloc[test_inds]
            data['orders']['train'], data['orders']['valid'], data['orders']['test'] = data['orders']['train'][train_inds], \
                data['orders']['train'][valid_inds], data['orders']['train'][test_inds]
            data['batches']['train'], data['batches']['valid'], data['batches']['test'] = data['batches']['train'][train_inds], \
                data['batches']['train'][valid_inds], data['batches']['train'][test_inds]
            data['cats']['train'], data['cats']['valid'], data['cats']['test'] = data['cats']['train'][train_inds], data['cats']['train'][
                valid_inds], data['cats']['train'][test_inds]

        else:
            meta = pd.read_csv(
                f"{path}/sample_annotation_AgingMice.csv", sep=","
            )
            meta_names = meta.loc[:, 'FullRunName']
            meta_labels = meta.loc[:, 'Diet']
            meta_batch = meta.loc[:, 'MS_batch']
            meta_order = meta.loc[:, 'order']
            matrix = pd.read_csv(
                f"{path}/proteome_log_AgingMice.csv", sep=",", index_col=0
            )
            names = matrix.columns
            pos = np.concatenate([
                np.argwhere(name == names)
                for i, name in enumerate(meta_names) if name in names.tolist()
            ]).squeeze()

            samples_to_keep = np.argwhere(meta_labels.values != 'CDHFD').reshape(-1)

            def impute_zero(peak):
                zero_mask = peak == 0
                if zero_mask.any():
                    new_x = peak.copy()
                    impute_value = peak.loc[~zero_mask].min()
                    new_x[zero_mask] = impute_value / 2
                    return new_x
                return peak

            matrix = matrix.fillna(0).iloc[:, pos].T.iloc[samples_to_keep]
            # if not args.zinb:
            #     matrix = matrix.apply(impute_zero, axis=0)
            if args.remove_zeros:
                mask1 = (matrix == 0).mean(axis=0) < 0.1
                matrix = matrix.loc[:, mask1]
            data['inputs'][group] = matrix
            data['names'][group] = meta_names.iloc[samples_to_keep]  # .iloc[meta_pos]
            data['labels'][group] = meta_labels.iloc[samples_to_keep].values  # .iloc[meta_pos].to_numpy()
            data['batches'][group] = meta_batch.iloc[samples_to_keep].values
            data['orders'][group] = meta_order.iloc[samples_to_keep].values
            data['meta'][group] = data['inputs'][group].iloc[:, :2]

            unique_labels = get_unique_labels(data['labels'][group])
            data['cats'][group] = np.array(
                [np.where(x == unique_labels)[0][0] for i, x in enumerate(data['labels'][group])])

    for key in list(data['names'].keys()):
        data['sets'][key] = np.array([key for _ in data['names'][key]])
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
    for group in ['train', 'valid', 'test', 'all']:
        data['batches'][group] = np.array([np.argwhere(unique_batches == x)[0][0] for x in data['batches'][group]])

    return data, unique_labels, unique_batches


def get_data(path, args, seed=42):
    """

    Args:
        path: Path where the csvs can be loaded. The folder designated by path needs to contain at least
                   one file named train_inputs.csv (when using --use_valid=0 and --use_test=0). When using
                   --use_valid=1 and --use_test=1, it must also contain valid_inputs.csv and test_inputs.csv.

    Returns:
        data
    """
    data = {}
    # batch_cols = args.batch_columns
    unique_labels = np.array([])
    for info in ['inputs', 'meta', 'names', 'labels', 'cats', 'batches', 'orders', 'sets']:
        data[info] = {}
        for group in ['all', 'train', 'test', 'valid']:
            data[info][group] = np.array([])
    for group in ['train', 'valid']:
        # print('GROUP:', group)
        if group == 'valid':
            if args.groupkfold:
                skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
                train_nums = np.arange(0, len(data['labels']['train']))
                # Remove samples from unwanted batches
                splitter = skf.split(train_nums, data['labels']['train'], data['batches']['train'])

            else:
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
                train_nums = np.arange(0, len(data['labels']['train']))
                splitter = skf.split(train_nums, data['labels']['train'])

            _, valid_inds = splitter.__next__()
            _, test_inds = splitter.__next__()
            train_inds = [x for x in train_nums if x not in np.concatenate((valid_inds, test_inds))]

            data['inputs']['train'], data['inputs']['valid'], data['inputs']['test'] = data['inputs']['train'].iloc[train_inds], \
                data['inputs']['train'].iloc[valid_inds], data['inputs']['train'].iloc[test_inds]
            data['meta']['train'], data['meta']['valid'], data['meta']['test'] = data['meta']['train'].iloc[train_inds], \
                data['meta']['train'].iloc[valid_inds], data['meta']['train'].iloc[test_inds]
            data['labels']['train'], data['labels']['valid'], data['labels']['test'] = data['labels']['train'][train_inds], \
                data['labels']['train'][valid_inds], data['labels']['train'][test_inds]
            data['names']['train'], data['names']['valid'], data['names']['test'] = data['names']['train'].iloc[train_inds], \
                data['names']['train'].iloc[valid_inds], data['names']['train'].iloc[test_inds]
            data['orders']['train'], data['orders']['valid'], data['orders']['test'] = data['orders']['train'][train_inds], \
                data['orders']['train'][valid_inds], data['orders']['train'][test_inds]
            data['batches']['train'], data['batches']['valid'], data['batches']['test'] = data['batches']['train'][train_inds], \
                data['batches']['train'][valid_inds], data['batches']['train'][test_inds]
            data['cats']['train'], data['cats']['valid'], data['cats']['test'] = data['cats']['train'][train_inds], data['cats']['train'][
                valid_inds], data['cats']['train'][test_inds]

            if args.pool:
                if args.groupkfold:
                    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
                    train_nums_pool = np.arange(0, len(data['labels']['train_pool']))
                    pool_splitter = skf.split(train_nums_pool, data['labels']['train_pool'],
                                                    data['batches']['train_pool'])

                else:
                    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
                    train_nums_pool = np.arange(0, len(data['labels']['train_pool']))
                    pool_splitter = skf.split(train_nums_pool, data['labels']['train_pool'])

                _, valid_inds = pool_splitter.__next__()
                _, test_inds = pool_splitter.__next__()
                train_inds = [x for x in train_nums_pool if x not in np.concatenate((valid_inds, test_inds))]
                data['inputs']['train_pool'], data['inputs']['valid_pool'], data['inputs']['test_pool'], = data['inputs']['train_pool'].iloc[train_inds], \
                    data['inputs']['train_pool'].iloc[valid_inds], data['inputs']['train_pool'].iloc[test_inds]
                data['meta']['train_pool'], data['meta']['valid_pool'], data['meta']['test_pool'], = data['meta']['train_pool'].iloc[train_inds], \
                    data['meta']['train_pool'].iloc[valid_inds], data['meta']['train_pool'].iloc[test_inds]
                data['labels']['train_pool'], data['labels']['valid_pool'], data['labels']['test_pool'], = data['labels']['train_pool'][train_inds], \
                    data['labels']['train_pool'][valid_inds], data['labels']['train_pool'][test_inds]
                data['names']['train_pool'], data['names']['valid_pool'], data['names']['test_pool'], = data['names']['train_pool'][train_inds], \
                    data['names']['train_pool'][valid_inds], data['names']['train_pool'][test_inds]
                data['orders']['train_pool'], data['orders']['valid_pool'], data['orders']['test_pool'], = data['orders']['train_pool'][train_inds], \
                    data['orders']['train_pool'][valid_inds], data['orders']['train_pool'][test_inds]
                data['batches']['train_pool'], data['batches']['valid_pool'], data['batches']['test_pool'], = data['batches']['train_pool'][train_inds], \
                    data['batches']['train_pool'][valid_inds], data['batches']['train_pool'][test_inds]
                data['cats']['train_pool'], data['cats']['valid_pool'], data['cats']['test_pool'], = data['cats']['train_pool'][train_inds], data['cats']['train_pool'][
                    valid_inds], data['cats']['train_pool'][test_inds]

        else:
            matrix = pd.read_csv(
                f"{path}/{args.csv_file}", sep=","
            )
            names = matrix.iloc[:, 0]
            labels = matrix.iloc[:, 1]
            batches = matrix.iloc[:, 2]
            unique_batches = batches.unique()
            batches = np.stack([np.argwhere(x == unique_batches).squeeze() for x in batches])
            orders = np.array([0 for _ in batches])
            matrix = matrix.iloc[:, 3:].fillna(0)
            if args.remove_zeros:
                mask1 = (matrix == 0).mean(axis=0) < 0.1
                matrix = matrix.loc[:, mask1]
            if args.log1p:
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

            # data['labels'][group] = np.array([x.split('-')[0] for i, x in enumerate(data['labels'][group])])
            unique_labels = get_unique_labels(data['labels'][group])
            data['cats'][group] = data['labels'][group]

            if args.pool:
                pool_pos = [i for i, name in enumerate(names.values.flatten()) if 'QC' in name]
                data['inputs'][f"{group}_pool"] = matrix.iloc[pool_pos]
                data['names'][f"{group}_pool"] = np.array([f'pool_{i}' for i, _ in enumerate(pool_pos)])
                data['labels'][f"{group}_pool"] = np.array([f'pool' for _ in pool_pos])
                data['batches'][f"{group}_pool"] = batches[pool_pos]

                # This is juste to make the pipeline work. Meta should be 0 for the amide dataset
                data['meta'][f"{group}_pool"] = data['inputs'][f"{group}_pool"].iloc[:, :2]
                data['orders'][f"{group}_pool"] = orders[pool_pos]
                data['cats'][f"{group}_pool"] = np.array(
                    [len(np.unique(data['labels'][group])) for _ in batches[pool_pos]])

                data['labels'][group] = np.array([x.split('-')[0] for i, x in enumerate(data['labels'][group])])
                unique_labels = np.concatenate((get_unique_labels(data['labels'][group]), np.array(['pool'])))
            data['cats'][group] = np.array(
                [np.where(x == unique_labels)[0][0] for i, x in enumerate(data['labels'][group])])

    for key in list(data['names'].keys()):
        data['sets'][key] = np.array([key for _ in data['names'][key]])
        # print(key, data['sets'][key])
    if not args.pool:
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
        for group in ['train', 'valid', 'test', 'all']:
            data['batches'][group] = np.array([np.argwhere(unique_batches == x)[0][0] for x in data['batches'][group]])
    else:
        # print('POOL!!')
        for key in list(data.keys()):
            # print('key', key)
            if key in ['inputs', 'meta']:
                data[key]['all'] = pd.concat((
                    data[key]['train'], data[key]['valid'], data[key]['test'],
                    data[key]['train_pool'], data[key]['valid_pool'], data[key]['test_pool'],
                ), 0)
                data[key]['all_pool'] = pd.concat((
                    data[key]['train_pool'], data[key]['valid_pool'], data[key]['test_pool'],
                ), 0)
            else:
                data[key]['all'] = np.concatenate((
                    data[key]['train'], data[key]['valid'], data[key]['test'],
                    data[key]['train_pool'], data[key]['valid_pool'], data[key]['test_pool'],
                ), 0)
                data[key]['all_pool'] = np.concatenate((
                    data[key]['train_pool'], data[key]['valid_pool'], data[key]['test_pool'],
                ), 0)

        unique_batches = np.unique(data['batches']['all'])
        for group in ['train', 'valid', 'test', 'train_pool', 'valid_pool', 'test_pool', 'all', 'all_pool']:
            data['batches'][group] = np.array([np.argwhere(unique_batches == x)[0][0] for x in data['batches'][group]])

    return data, unique_labels, unique_batches


def get_bacteria_images(path, args, seed=42):
    """

    Args:
        path: Path where the csvs can be loaded. The folder designated by path needs to contain at least
                   one file named train_inputs.csv (when using --use_valid=0 and --use_test=0). When using
                   --use_valid=1 and --use_test=1, it must also contain valid_inputs.csv and test_inputs.csv.

    Returns:
        data
    """
    data = {}
    unique_labels = np.array([])
    for info in ['inputs', 'meta', 'names', 'labels', 'cats', 'batches', 'orders', 'sets']:
        data[info] = {}
        for group in ['all', 'train', 'test', 'valid']:
            data[info][group] = np.array([])
    for group in ['train', 'valid', 'test']:
        if group == 'valid':
            if args.groupkfold:
                skf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)
                train_nums = np.arange(0, len(data['labels']['train']))
                # Remove samples from unwanted batches
                train_inds, valid_inds = skf.split(train_nums, data['labels']['train'],
                                                   data['batches']['train']).__next__()
            else:
                skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
                train_nums = np.arange(0, len(data['labels']['train']))
                train_inds, valid_inds = skf.split(train_nums, data['labels']['train']).__next__()

            # matrix = matrix.fillna(0).iloc[:, pos].T.iloc[samples_to_keep]
            # if not args.zinb:
            # matrix = matrix.apply(impute_zero, axis=0)
            if args.remove_zeros:
                mask1 = (matrix == 0).mean(axis=0) < 0.1
                matrix = matrix.loc[:, mask1]
            data['inputs']['valid'], data['inputs']['train'] = data['inputs']['train'][valid_inds], data['inputs']['train'][train_inds]
            data['names']['valid'], data['names']['train'] = data['names']['train'][valid_inds], data['names']['train'][train_inds]
            data['labels']['valid'], data['labels']['train'] = data['labels']['train'][valid_inds], data['labels']['train'][train_inds]  # .iloc[meta_pos].to_numpy()
            data['batches']['valid'], data['batches']['train'] = data['batches']['train'][valid_inds], data['batches']['train'][train_inds]
            data['orders']['valid'], data['orders']['train'] = data['orders']['train'][valid_inds], data['orders']['train'][train_inds]
            data['meta']['valid'], data['meta']['train'] = data['inputs'][group], data['inputs']['train']
            data['sets']['valid'], data['sets']['train'] = data['sets']['train'][valid_inds], data['sets']['train'][train_inds]
            data['sets']['valid'] = np.array(['valid' for _ in data['names']['valid']])

            unique_labels1 = get_unique_labels(data['labels'][group])

        elif group == 'test':
            if args.groupkfold:
                skf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)
                train_nums = np.arange(0, len(data['labels']['train']))
                # Remove samples from unwanted batches
                train_inds, valid_inds = skf.split(train_nums, data['labels']['train'],
                                                   data['batches']['train']).__next__()
            else:
                skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
                train_nums = np.arange(0, len(data['labels']['train']))
                train_inds, valid_inds = skf.split(train_nums, data['labels']['train']).__next__()

            # matrix = matrix.fillna(0).iloc[:, pos].T.iloc[samples_to_keep]
            # if not args.zinb:
            # matrix = matrix.apply(impute_zero, axis=0)
            if args.remove_zeros:
                mask1 = (matrix == 0).mean(axis=0) < 0.1
                matrix = matrix.loc[:, mask1]
            # This is juste to make the pipeline work. Meta should be 0 for the amide dataset
            data['inputs']['test'], data['inputs']['train'] = data['inputs']['train'][valid_inds], data['inputs']['train'][train_inds]
            data['names']['test'], data['names']['train'] = data['names']['train'][valid_inds], data['names']['train'][train_inds]
            data['labels']['test'], data['labels']['train'] = data['labels']['train'][valid_inds], data['labels']['train'][train_inds]  # .iloc[meta_pos].to_numpy()
            data['batches']['test'], data['batches']['train'] = data['batches']['train'][valid_inds], data['batches']['train'][train_inds]
            data['orders']['test'], data['orders']['train'] = data['orders']['train'][valid_inds], data['orders']['train'][train_inds]
            data['meta']['test'], data['meta']['train'] = data['inputs'][group], data['inputs']['train']
            data['sets']['test'], data['sets']['train'] = data['sets']['train'][valid_inds], data['sets']['train'][train_inds]
            data['sets']['test'] = np.array(['test' for _ in data['names']['test']])

            unique_labels2 = get_unique_labels(data['labels'][group])

        else:
            process = MSCSV(path, args.scaler, new_size=32)
            pool = multiprocessing.Pool(multiprocessing.cpu_count() - 1)
            data_images = pool.map(process.process, range(process.__len__()))
            images, labels, batches, plates, names  = [x[0] for x in data_images], pd.Series(
                [x[1] for x in data_images]), pd.Series(
                [x[2] for x in data_images]), pd.Series([x[3] for x in data_images]), pd.Series([x[4] for x in data_images])
            pool.close()
            pool.join()
            pool.terminate()
            if args.log1p:
                images = np.log1p(images)
            del pool, data_images

            if args.remove_zeros:
                mask1 = (matrix == 0).mean(axis=0) < 0.1
                matrix = matrix.loc[:, mask1]
            data['inputs'][group] = np.stack(images)
            data['names'][group] = names.values
            data['labels'][group] = labels.values  # .iloc[meta_pos].to_numpy()
            data['batches'][group] = batches.values
            data['orders'][group] = plates.values
            data['meta'][group] = data['inputs'][group]
            data['sets'][group] = np.array([group for _ in data['names'][group]])

            pos = [i for i, name in enumerate(data['labels'][group].flatten()) if 'pool' not in data['labels'][group][i]]
            data['names'][group] = data['names'][group][pos]
            data['labels'][group] = data['labels'][group][pos]
            data['batches'][group] = data['batches'][group][pos]
            data['meta'][group] = data['meta'][group][pos]
            data['orders'][group] = data['orders'][group][pos]
            data['inputs'][group] = data['inputs'][group][pos]
            data['sets'][group] = data['sets'][group][pos]
            unique_labels3 = get_unique_labels(data['labels'][group])

    # Testing using the smallest number of samples for training
    # data['inputs']['test'], data['inputs']['train'] = data['inputs']['train'], data['inputs']['test']
    # data['names']['test'], data['names']['train'] = data['names']['train'], data['names']['test']
    # data['labels']['test'], data['labels']['train'] = data['labels']['train'], data['labels']['test']
    # data['batches']['test'], data['batches']['train'] = data['batches']['train'], data['batches']['test']
    # data['orders']['test'], data['orders']['train'] = data['orders']['train'], data['orders']['test']
    # data['meta']['test'], data['meta']['train'] = data['inputs'][group], data['inputs']['test']

    for key in list(data.keys()):
        data[key]['all'] = np.concatenate((
            data[key]['train'], data[key]['valid'], data[key]['test']
        ), 0)

    unique_labels = np.unique(np.concatenate((unique_labels1, unique_labels2, unique_labels3)))
    unique_batches = np.unique(data['batches']['all'])
    # must be split based on batches, but batches should be plates
    for group in ['train', 'valid', 'test', 'all']:
        data['batches'][group] = np.array([np.argwhere(unique_batches == x)[0][0] for x in data['batches'][group]])
    for group in ['train', 'valid', 'test', 'all']:
        data['cats'][group] = np.array(
            [np.where(x == unique_labels)[0][0] for i, x in enumerate(data['labels'][group])])

    # If we also load blanks in the samples, it should help a lot
    # I will put half the blanks from the valid and test sets in the train set.
    # In production, we will only need to have blanks to process with the samples
    for group in ['valid', 'test']:
        blks_pos = np.argwhere(data['labels'][group] == 'blk').flatten().tolist()
        blks_to_move = random.sample(blks_pos, int(len(blks_pos) / 2))
        blks_not_to_move = [x for x in blks_pos if x not in blks_to_move]
        not_to_move = np.argwhere(data['labels'][group] != 'blk').flatten().tolist() + blks_not_to_move
        data['batches']['train'], data['batches'][group] = np.concatenate(
            (data['batches']['train'], data['batches'][group][blks_to_move])), data['batches'][group][not_to_move],
        data['inputs']['train'], data['inputs'][group] = np.concatenate(
            (data['inputs']['train'], data['inputs'][group][blks_to_move])), data['inputs'][group][not_to_move]
        data['meta']['train'], data['meta'][group] = np.concatenate(
            (data['meta']['train'], data['meta'][group][blks_to_move])), data['meta'][group][not_to_move]
        data['cats']['train'], data['cats'][group] = np.concatenate(
            (data['cats']['train'], data['cats'][group][blks_to_move])), data['cats'][group][not_to_move]
        data['labels']['train'], data['labels'][group] = np.concatenate(
            (data['labels']['train'], data['labels'][group][blks_to_move])), data['labels'][group][not_to_move]
        data['orders']['train'], data['orders'][group] = np.concatenate(
            (data['orders']['train'], data['orders'][group][blks_to_move])), data['orders'][group][not_to_move]
        data['names']['train'], data['names'][group] = np.concatenate(
            (data['names']['train'], data['names'][group][blks_to_move])), data['names'][group][not_to_move]
        data['sets']['train'], data['sets'][group] = np.concatenate(
            (data['sets']['train'], data['sets'][group][blks_to_move])), data['sets'][group][not_to_move]

    return data, unique_labels, unique_batches


def get_bacteria_images_ms2(path, args, seed=42):
    """

    Args:
        path: Path where the csvs can be loaded. The folder designated by path needs to contain at least
                   one file named train_inputs.csv (when using --use_valid=0 and --use_test=0). When using
                   --use_valid=1 and --use_test=1, it must also contain valid_inputs.csv and test_inputs.csv.

    Returns:
        data
    """
    data = {}
    unique_labels = np.array([])
    for info in ['inputs', 'meta', 'names', 'labels', 'cats', 'batches', 'orders', 'sets']:
        data[info] = {}
        for group in ['all', 'train', 'test', 'valid']:
            data[info][group] = np.array([])
    for group in ['train', 'valid', 'test']:
        if group == 'valid':
            if args.groupkfold:
                skf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)
                train_nums = np.arange(0, len(data['labels']['train']))
                # Remove samples from unwanted batches
                train_inds, valid_inds = skf.split(train_nums, data['labels']['train'],
                                                   data['batches']['train']).__next__()
            else:
                skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
                train_nums = np.arange(0, len(data['labels']['train']))
                train_inds, valid_inds = skf.split(train_nums, data['labels']['train']).__next__()

            # matrix = matrix.fillna(0).iloc[:, pos].T.iloc[samples_to_keep]
            # if not args.zinb:
            # matrix = matrix.apply(impute_zero, axis=0)
            if args.remove_zeros:
                mask1 = (matrix == 0).mean(axis=0) < 0.1
                matrix = matrix.loc[:, mask1]
            data['inputs']['valid'], data['inputs']['train'] = data['inputs']['train'][valid_inds], data['inputs']['train'][train_inds]
            data['names']['valid'], data['names']['train'] = data['names']['train'][valid_inds], data['names']['train'][train_inds]
            data['labels']['valid'], data['labels']['train'] = data['labels']['train'][valid_inds], data['labels']['train'][train_inds]  # .iloc[meta_pos].to_numpy()
            data['batches']['valid'], data['batches']['train'] = data['batches']['train'][valid_inds], data['batches']['train'][train_inds]
            data['orders']['valid'], data['orders']['train'] = data['orders']['train'][valid_inds], data['orders']['train'][train_inds]
            data['meta']['valid'], data['meta']['train'] = data['inputs'][group], data['inputs']['train']
            data['sets']['valid'], data['sets']['train'] = data['sets']['train'][valid_inds], data['sets']['train'][train_inds]
            data['sets']['valid'] = np.array(['valid' for _ in data['names']['valid']])

            unique_labels1 = get_unique_labels(data['labels'][group])

        elif group == 'test':
            if args.groupkfold:
                skf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)
                train_nums = np.arange(0, len(data['labels']['train']))
                # Remove samples from unwanted batches
                train_inds, valid_inds = skf.split(train_nums, data['labels']['train'],
                                                   data['batches']['train']).__next__()
            else:
                skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
                train_nums = np.arange(0, len(data['labels']['train']))
                train_inds, valid_inds = skf.split(train_nums, data['labels']['train']).__next__()

            # matrix = matrix.fillna(0).iloc[:, pos].T.iloc[samples_to_keep]
            # if not args.zinb:
            # matrix = matrix.apply(impute_zero, axis=0)
            if args.remove_zeros:
                mask1 = (matrix == 0).mean(axis=0) < 0.1
                matrix = matrix.loc[:, mask1]
            # This is juste to make the pipeline work. Meta should be 0 for the amide dataset
            data['inputs']['test'], data['inputs']['train'] = data['inputs']['train'][valid_inds], data['inputs']['train'][train_inds]
            data['names']['test'], data['names']['train'] = data['names']['train'][valid_inds], data['names']['train'][train_inds]
            data['labels']['test'], data['labels']['train'] = data['labels']['train'][valid_inds], data['labels']['train'][train_inds]  # .iloc[meta_pos].to_numpy()
            data['batches']['test'], data['batches']['train'] = data['batches']['train'][valid_inds], data['batches']['train'][train_inds]
            data['orders']['test'], data['orders']['train'] = data['orders']['train'][valid_inds], data['orders']['train'][train_inds]
            data['meta']['test'], data['meta']['train'] = data['inputs'][group], data['inputs']['train']
            data['sets']['test'], data['sets']['train'] = data['sets']['train'][valid_inds], data['sets']['train'][train_inds]
            data['sets']['test'] = np.array(['test' for _ in data['names']['test']])

            unique_labels2 = get_unique_labels(data['labels'][group])

        else:
            process = MS2CSV(path, args.scaler, new_size=32)
            pool = multiprocessing.Pool(multiprocessing.cpu_count() - 1)
            data_images = pool.map(process.process, range(process.__len__()))
            images, labels, batches, plates, names = [x[0] for x in data_images], pd.Series(
                [x[1] for x in data_images]), pd.Series(
                [x[2] for x in data_images]), pd.Series([x[3] for x in data_images]), pd.Series([x[4] for x in data_images])
            pool.close()
            pool.join()
            pool.terminate()
            if args.log1p:
                images = np.log1p(images)
            del pool, data_images

            if args.remove_zeros:
                mask1 = (matrix == 0).mean(axis=0) < 0.1
                matrix = matrix.loc[:, mask1]
            data['inputs'][group] = np.stack(images)
            data['names'][group] = names.values
            data['labels'][group] = labels.values  # .iloc[meta_pos].to_numpy()
            data['batches'][group] = batches.values
            data['orders'][group] = plates.values
            data['meta'][group] = data['inputs'][group]
            data['sets'][group] = np.array([group for _ in data['names'][group]])

            pos = [i for i, name in enumerate(data['labels'][group].flatten()) if 'pool' not in data['labels'][group][i]]
            data['names'][group] = data['names'][group][pos]
            data['labels'][group] = data['labels'][group][pos]
            data['batches'][group] = data['batches'][group][pos]
            data['meta'][group] = data['meta'][group][pos]
            data['orders'][group] = data['orders'][group][pos]
            data['inputs'][group] = data['inputs'][group][pos]
            data['sets'][group] = data['sets'][group][pos]
            unique_labels3 = get_unique_labels(data['labels'][group])

    # Testing using the smallest number of samples for training
    # data['inputs']['test'], data['inputs']['train'] = data['inputs']['train'], data['inputs']['test']
    # data['names']['test'], data['names']['train'] = data['names']['train'], data['names']['test']
    # data['labels']['test'], data['labels']['train'] = data['labels']['train'], data['labels']['test']
    # data['batches']['test'], data['batches']['train'] = data['batches']['train'], data['batches']['test']
    # data['orders']['test'], data['orders']['train'] = data['orders']['train'], data['orders']['test']
    # data['meta']['test'], data['meta']['train'] = data['inputs'][group], data['inputs']['test']

    for key in list(data.keys()):
        data[key]['all'] = np.concatenate((
            data[key]['train'], data[key]['valid'], data[key]['test']
        ), 0)

    unique_labels = np.unique(np.concatenate((unique_labels1, unique_labels2, unique_labels3)))
    unique_batches = np.unique(data['batches']['all'])
    # must be split based on batches, but batches should be plates
    for group in ['train', 'valid', 'test', 'all']:
        data['batches'][group] = np.array([np.argwhere(unique_batches == x)[0][0] for x in data['batches'][group]])
    for group in ['train', 'valid', 'test', 'all']:
        data['cats'][group] = np.array(
            [np.where(x == unique_labels)[0][0] for i, x in enumerate(data['labels'][group])])

    # If we also load blanks in the samples, it should help a lot
    # I will put half the blanks from the valid and test sets in the train set.
    # In production, we will only need to have blanks to process with the samples
    for group in ['valid', 'test']:
        blks_pos = np.argwhere(data['labels'][group] == 'blk').flatten().tolist()
        blks_to_move = random.sample(blks_pos, int(len(blks_pos) / 2))
        blks_not_to_move = [x for x in blks_pos if x not in blks_to_move]
        not_to_move = np.argwhere(data['labels'][group] != 'blk').flatten().tolist() + blks_not_to_move
        data['batches']['train'], data['batches'][group] = np.concatenate(
            (data['batches']['train'], data['batches'][group][blks_to_move])), data['batches'][group][not_to_move],
        data['inputs']['train'], data['inputs'][group] = np.concatenate(
            (data['inputs']['train'], data['inputs'][group][blks_to_move])), data['inputs'][group][not_to_move]
        data['meta']['train'], data['meta'][group] = np.concatenate(
            (data['meta']['train'], data['meta'][group][blks_to_move])), data['meta'][group][not_to_move]
        data['cats']['train'], data['cats'][group] = np.concatenate(
            (data['cats']['train'], data['cats'][group][blks_to_move])), data['cats'][group][not_to_move]
        data['labels']['train'], data['labels'][group] = np.concatenate(
            (data['labels']['train'], data['labels'][group][blks_to_move])), data['labels'][group][not_to_move]
        data['orders']['train'], data['orders'][group] = np.concatenate(
            (data['orders']['train'], data['orders'][group][blks_to_move])), data['orders'][group][not_to_move]
        data['names']['train'], data['names'][group] = np.concatenate(
            (data['names']['train'], data['names'][group][blks_to_move])), data['names'][group][not_to_move]
        data['sets']['train'], data['sets'][group] = np.concatenate(
            (data['sets']['train'], data['sets'][group][blks_to_move])), data['sets'][group][not_to_move]

    return data, unique_labels, unique_batches

