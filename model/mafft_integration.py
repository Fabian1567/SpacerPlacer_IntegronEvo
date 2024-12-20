import itertools
import json
import pickle
import subprocess
import os
import numpy as np
import platform

from sklearn.manifold import MDS
from sklearn.preprocessing import MinMaxScaler

from model.helpers import import_data
from additional_data.additional_scripts.model.helpers import raw_data_tools
from model.data_classes.crisprdata import CRISPRArray

# Excluded characters that are not allowed by mafft.
EXCLUDED_CHARACTERS = {'3e': 'ff', '3d': 'fe', '3c': 'fd',
                       '2d': 'fc', '20': 'fb', '0d': 'fa',
                       '0a': 'f9'}
REVERSED_EXCLUDED_CHARACTERS = {val: key for key, val in EXCLUDED_CHARACTERS.items()}

DEFAULT_MAFFT_OPTIONS = ['--text',
                         # '--genafpair',
                         '--localpair',
                         # '--sp',
                         # '--globalpair',
                         # '--hybridpair',
                         # '--leavegappyregion',
                         '--maxiterate', '1000',
                         # '--noscore',
                         '--lep', '0',
                         '--op', '0',
                         '--lop', '0',
                         '--lep', '0',
                         '--lexp', '0',
                         '--LOP', '0',
                         '--LEXP', '0',
                         '--quiet',
                         '--thread', '-1',  # multithreading -1 -> automatically choose
                         # '--threadit', '0',  # threaded iterative alignment has some randomness (this stops this)
                         ]

DEFAULT_MAFFT_OPTIONS_FFT = ['--text',
                             '--retree', '2',
                             # '--sp',
                             # '--globalpair',
                             # '--hybridpair',
                             # '--leavegappyregion',
                             '--maxiterate', '1000',
                             # '--noscore',
                             '--lep', '0',
                             '--op', '0',
                             '--lop', '0',
                             '--lep', '0',
                             '--lexp', '0',
                             '--LOP', '0',
                             '--LEXP', '0',
                             '--quiet',
                             '--thread', '1',  # '-1',  # multithreading -1 -> automatically choose
                             # '--threadit', '0',  # threaded iterative alignment has some randomness (this stops this)
                             ]

# probably don't need windows as extra case, need to try it out (exe make it more annoying?)
if platform.system() == 'Darwin':
    MAFFT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'mafft_scripts',
                              'mafft-mac')
# elif platform.system() == 'Windows':
#     MAFFT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'mafft_scripts',
#                               'mafft-win')
else:
    MAFFT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'mafft_scripts',
                              'mafft-linux64')


# MAFFT_PATH = '/usr/libexec/mafft'

def generate_pseudo_spacer_dna(alphabet, length=20):
    pseudo_dna = np.random.choice(list(alphabet), length, replace=True)
    return list(pseudo_dna)


def convert_spacer_arrays_to_random_pseudo_dna(ls_arrays, length=20):
    all_unique_spacers = set().union(*[set(a) for a in ls_arrays])
    dict_alphabets = dict()
    if len(all_unique_spacers) <= 58:
        tuples_4 = iter([range(a, b) for a, b in zip(range(5, 245), range(9, 249))])
        for a in all_unique_spacers:
            tuple_4 = next(tuples_4)
            alphabet = {str(tuple_4[0]), str(tuple_4[1]), str(tuple_4[2]), str(tuple_4[3]),
                        }
            dict_alphabets[a] = alphabet
    else:
        set_alphabets = set()
        considered_alphabet_range = list(range(5, 249))
        for i, a in enumerate(all_unique_spacers):
            while True:
                tuple_4 = tuple(np.random.choice(considered_alphabet_range, 4, replace=False))
                if tuple_4 not in set_alphabets:
                    dict_alphabets[a] = {str(tuple_4[0]), str(tuple_4[1]),
                                         str(tuple_4[2]), str(tuple_4[3]),
                                         }
                    set_alphabets.add(tuple_4)
                    break

    dict_unique_id_spacers = {a: generate_pseudo_spacer_dna(alph, length=length) for a, alph in dict_alphabets.items()}

    new_ls_arrays = []
    for array in ls_arrays:
        new_array = ['1', '2', '3', '4', '3', '2', '1']
        for v in array:
            new_array += dict_unique_id_spacers[v]
            new_array += ['1', '2', '3', '4', '3', '2', '1']
        new_ls_arrays.append(new_array)
    return new_ls_arrays, dict_unique_id_spacers, dict_alphabets, '1234321'


def do_mds(matrix):
    # Perform MDS
    embedding = MDS(n_components=2, dissimilarity='precomputed')
    X_transformed = embedding.fit_transform(matrix)

    # Scale the coordinates to fit within the range [0, 25]
    scaler = MinMaxScaler(feature_range=(0, 25))
    X_transformed_scaled = scaler.fit_transform(X_transformed)

    # Map scaled coordinates to integers
    X_transformed_integer = np.floor(X_transformed_scaled).astype(int)

    return X_transformed_integer


def convert_spacer_arrays_to_random_pseudo_dna_cluster(ls_arrays, reverse_cluster_map, matrices, length=22):
    dict_alphabets = dict()
    dict_unique_id_spacers = {}

    set_alphabets = set()
    considered_alphabet_range = list(range(9, 197))
    mapping_one = np.arange(223, 249)
    mapping_two = np.arange(197, 223)

    for key, val in reverse_cluster_map.items():
        current_alphabet = []
        while True:
            tuple_4 = tuple(np.random.choice(considered_alphabet_range, 4, replace=False))
            if tuple_4 not in set_alphabets:
                current_alphabet = {str(tuple_4[0]), str(tuple_4[1]),
                                    str(tuple_4[2]), str(tuple_4[3]),
                                    }
                set_alphabets.add(tuple_4)
                break

        if len(val) == 1:
            dict_unique_id_spacers[str(val[0])] = generate_pseudo_spacer_dna(current_alphabet, length=length)
        else:
            mds = do_mds(matrices[key - 1])
            base_seq = generate_pseudo_spacer_dna(current_alphabet, length=22)
            repeat_length = 1  # Length of the repeat
            for i in range(len(val)):
                temp = []
                coords = mds[i]
                #
                # # Calculate the starting position for the insert
                # middle_start = len(temp) // 2 - (repeat_length // 2)
                #
                # # Insert repeats into the middle of the sequence
                # for j in range(repeat_length):
                #     temp.insert(middle_start + j, str(mapping_one[coords[0]]))  # First repeat
                # for j in range(repeat_length):
                #     temp.insert(middle_start + repeat_length + j, str(mapping_two[coords[1]]))

                temp.append(str(mapping_one[coords[0]]))
                temp.append(str(mapping_two[coords[1]]))

                temp.append(str(mapping_one[coords[0]]))
                temp.append(str(mapping_two[coords[1]]))

                dict_unique_id_spacers[str(val[i])] = temp

    new_ls_arrays = []
    for array in ls_arrays:
        new_array = ['1', '2', '3', '4', '3', '2', '1']
        for v in array:
            new_array += dict_unique_id_spacers[v]
            new_array += ['1', '2', '3', '4', '3', '2', '1']
        new_ls_arrays.append(new_array)
    return new_ls_arrays, dict_unique_id_spacers, dict_alphabets, '1234321'


def convert_pseudo_dna_to_spacer_arrays(ls_arrays, dict_unique_id_spacers, dict_alphabets, repeat):
    values_in_repeat = set(repeat)
    alignment_length = len(ls_arrays[0])
    nb_arrays = len(ls_arrays)
    new_ls_arrays = [[] for _ in range(nb_arrays)]

    dict_spacers_unique_id = {''.join(value): key for key, value in dict_unique_id_spacers.items()}
    current_frag = ['' for _ in range(nb_arrays)]
    frag_memory = [[] for _ in range(nb_arrays)]
    for j in range(alignment_length):
        for i in range(nb_arrays):
            a = ls_arrays[i][j]
            if a == '--':
                continue
            elif a in values_in_repeat:
                current_frag[i] += a
            else:
                current_frag[i] += a

        for i in range(nb_arrays):
            if current_frag[i] == repeat:
                frag_memory[i].append('R')
                current_frag[i] = ''
            elif current_frag[i] in dict_spacers_unique_id:
                frag_memory[i].append(dict_spacers_unique_id[current_frag[i]])
                current_frag[i] = ''

        if any((len(m) >= 2 for m in frag_memory)):
            for idx, m in enumerate(frag_memory):
                if len(m) >= 2:
                    for k in range(2):
                        spacer_or_repeat = m.pop(0)
                        if spacer_or_repeat != 'R':
                            new_ls_arrays[idx].append(spacer_or_repeat)
                else:
                    new_ls_arrays[idx].append('--')
    return new_ls_arrays


def array_to_hex(ls_arrays):
    output = []
    unique_hx_val = []
    for array in ls_arrays:
        inner = []
        for x in array:
            a = hex(int(x))[2:]
            if len(a) == 1:
                a = '0' + a
            if a in EXCLUDED_CHARACTERS:
                a = EXCLUDED_CHARACTERS[a]
            unique_hx_val.append('0x' + a)
            inner.append(a)
        output.append(inner)
    unique_hx_val = np.unique(unique_hx_val)
    return output, unique_hx_val


def pseudo_dna_array_to_hex(ls_arrays):
    new_ls_arrays = []
    unique_val = set().union(*[set(array) for array in ls_arrays])
    unique_hx_val = dict()
    for val in unique_val:
        hex_val = hex(int(val))[2:]
        if len(hex_val) == 1:
            hex_val = '0' + hex_val
        if hex_val in EXCLUDED_CHARACTERS:
            hex_val = EXCLUDED_CHARACTERS[hex_val]
        unique_hx_val[str(val)] = '0x' + hex_val

    for array in ls_arrays:
        new_array = []
        for a in array:
            new_array += [*str(a)]
        new_array = [unique_hx_val[a] for a in array]
        new_ls_arrays.append(new_array)
    return new_ls_arrays, list(unique_hx_val.values())


def hex_to_array(ls_arrays):
    ls_arrays = [[int(x, base=16) for x in array] for array in ls_arrays]
    return ls_arrays


def write_matrixfile(unique_hex_val, path, file_name, match=10, mismatch=-1000):
    str_match = str(match)
    str_mismatch = str(mismatch)
    with open(os.path.join(path, file_name + '.txt'), 'w') as f:
        for (val_0, val_1) in itertools.combinations_with_replacement(unique_hex_val, 2):
            if val_0 == val_1:
                f.write(' '.join([val_0, val_1, str_match]))
            else:
                f.write(' '.join([val_0, val_1, str_mismatch]))
            f.write('\n')
    return os.path.join(path, file_name + '.txt')


def write_matrixfile_cluster(unique_hex_val, path, file_name, cluster_map, distances, hx_to_id_dict, cluster_score,
                             match=10,
                             mismatch=-1000):
    str_match = str(match)
    str_mismatch = str(mismatch)
    with open(os.path.join(path, file_name + '.txt'), 'w') as f:
        for (val_0, val_1) in itertools.combinations_with_replacement(unique_hex_val, 2):
            if val_0 == val_1:
                f.write(' '.join([val_0, val_1, str_match]))
            else:
                if cluster_score == 'exp':
                    dist = get_distance_exp(hx_to_id_dict[val_0[2:]], hx_to_id_dict[val_1[2:]], cluster_map, distances)
                else:
                    dist = get_distance(hx_to_id_dict[val_0[2:]], hx_to_id_dict[val_1[2:]], cluster_map, distances)

                if dist is None:
                    f.write(' '.join([val_0, val_1, str_mismatch]))
                else:
                    f.write(' '.join([val_0, val_1, str(dist)]))
            f.write('\n')
    return os.path.join(path, file_name + '.txt')


def get_distance(val_0, val_1, cluster_map, distances):
    # Check if both sequences belong to the same cluster
    if cluster_map.get(val_0) == cluster_map.get(val_1):
        cluster_idx = cluster_map[val_0] - 1
        cluster_distances = distances[cluster_idx]
        max_dist = max([dist for sub_dict in cluster_distances.values() for dist in sub_dict.values()])

        # Ensure the distance is stored only in one direction (seq1 < seq2)
        if val_0 < val_1:
            dist = cluster_distances.get(val_0, {}).get(val_1)
            return 10 - (dist / max_dist) * (10 - 5)
        else:
            dist = cluster_distances.get(val_1, {}).get(val_0)
            return 10 - (dist / max_dist) * (10 - 5)
    else:
        return None  # seq1 and seq2 are in different clusters


def get_distance_exp(val_0, val_1, cluster_map, distances):
    # Check if both sequences belong to the same cluster
    if cluster_map.get(val_0) == cluster_map.get(val_1):
        cluster_idx = cluster_map[val_0] - 1
        cluster_distances = distances[cluster_idx]
        max_dist = max([dist for sub_dict in cluster_distances.values() for dist in sub_dict.values()])

        # Ensure the distance is stored only in one direction (seq1 < seq2)
        if val_0 < val_1:
            dist = cluster_distances.get(val_0, {}).get(val_1)
        else:
            dist = cluster_distances.get(val_1, {}).get(val_0)

        scaled_value = 10 * ((max_dist - dist) / max_dist) ** 2
        return scaled_value
    else:
        return None  # seq1 and seq2 are in different clusters


def write_matrixfile_cluster_pseudo(unique_hex_val, path, file_name, match=10, mismatch=-1000):
    str_match = str(match)
    str_mismatch = str(mismatch)

    mapping_one = np.arange(223, 249)
    hex_dict_one = {hex(value): value for value in mapping_one}
    mapping_two = np.arange(197, 223)
    hex_dict_two = {hex(value): value for value in mapping_two}

    with open(os.path.join(path, file_name + '.txt'), 'w') as f:
        for (val_0, val_1) in itertools.combinations_with_replacement(unique_hex_val, 2):
            if val_0 in hex_dict_one and val_1 in hex_dict_one:
                dist = 1000 - (abs(hex_dict_one[val_0] - hex_dict_one[val_1]) / 25) * 1000
                f.write(' '.join([val_0, val_1, str(dist)]))
            elif val_0 in hex_dict_two and val_1 in hex_dict_two:
                dist = 1000 - (abs(hex_dict_two[val_0] - hex_dict_two[val_1]) / 25) * 1000
                f.write(' '.join([val_0, val_1, str(dist)]))
            elif val_0 == val_1:
                f.write(' '.join([val_0, val_1, str_match]))
            else:
                f.write(' '.join([val_0, val_1, str_mismatch]))
            f.write('\n')
    return os.path.join(path, file_name + '.txt')


def write_fasta(ls_arrays, array_names, path, file_name):
    save_path = os.path.join(path, file_name + '.txt')
    # print('ha', ls_arrays, array_names)
    with open(save_path, 'w') as f:
        for i in range(len(ls_arrays)):
            f.write('>' + array_names[i])
            f.write('\n')
            f.write(' '.join(ls_arrays[i]))
            f.write('\n')
    return save_path


def convert_mafft_output(file_path):
    ls_arrays = []
    ls_names = []
    with open(file_path, 'r') as f:
        output = f.readlines()
        for o in output:
            if o[0] == '>':
                ls_names.append(o.strip('>\n'))
                converted_array = []
                ls_arrays.append(converted_array)
            else:
                array = o.strip('\n')
                ls_array = array.split(' ')

                for pos in ls_array:
                    if pos == '--':
                        converted_array.append(pos)
                    elif pos == '':
                        continue
                    else:
                        if pos in REVERSED_EXCLUDED_CHARACTERS:
                            pos = REVERSED_EXCLUDED_CHARACTERS[pos]
                        converted_array.append(str(int(pos, 16)))
    return ls_arrays, ls_names


def align_crispr_groups(work_path, dict_crispr_groups, cluster_json_path, cluster_score, mafft_options=None,
                        logger=None, seed=None):
    if not os.path.exists(work_path):
        os.makedirs(work_path)

    for group_name, crispr_group in dict_crispr_groups.items():

        cluster_map, distances = None, None
        if cluster_json_path is not None:
            cluster_map, distances = load_cluster_info(cluster_json_path, group_name)

        dict_arrays_as_list = crispr_group.get_arrays_as_lists()
        ls_names = list(dict_arrays_as_list.keys())
        ls_arrays = list(dict_arrays_as_list.values())

        unique_spacers = set().union(*[set(a) for a in ls_arrays])
        if len(unique_spacers) > 248:
            pseudo_dna = True
            mafft_options = DEFAULT_MAFFT_OPTIONS_FFT if mafft_options is None else mafft_options
        else:
            pseudo_dna = False
            mafft_options = DEFAULT_MAFFT_OPTIONS if mafft_options is None else mafft_options

        if seed is not None:
            mafft_options.append('--randomseed')
            mafft_options.append(str(seed))

        if len(ls_arrays) > 1000:
            if logger is not None:
                logger.warning('Group {} contains {} arrays. This might take a while.'.format(group_name,
                                                                                              len(ls_arrays)))
                logger.warning('Consider performing the alignment on a subset of the data.')
            else:
                print('Consider performing the alignment on a subset of the data.')
                print('The current group contains {} arrays.'.format(len(ls_arrays)))

        # Do I want this?
        if not ls_arrays:
            raise ValueError('No arrays in group {}. Mafft might not have finished.'.format(group_name))

        if pseudo_dna:
            if cluster_json_path is not None:
                reverse_cluster_map = compute_reverse_cluster_map(cluster_map)
                distance_matrices = compute_distance_matrices(distances)

                ls_renamed_arrays, dict_unique_id_spacers, \
                    dict_alphabets, repeat = convert_spacer_arrays_to_random_pseudo_dna_cluster(ls_arrays,
                                                                                                reverse_cluster_map,
                                                                                                distance_matrices,
                                                                                                length=22)

                ls_hx_arrays, unique_hx_val = pseudo_dna_array_to_hex(ls_renamed_arrays)

                mx_path = write_matrixfile_cluster_pseudo(unique_hx_val, work_path, group_name + '_mx')

            else:
                ls_renamed_arrays, dict_unique_id_spacers, \
                    dict_alphabets, repeat = convert_spacer_arrays_to_random_pseudo_dna(ls_arrays, length=20)

                ls_hx_arrays, unique_hx_val = pseudo_dna_array_to_hex(ls_renamed_arrays)

                mx_path = write_matrixfile(unique_hx_val, work_path, group_name + '_mx')
        else:
            ls_renamed_arrays, dict_renaming, dict_reverse_renaming = rename_spacers_for_mafft(ls_arrays)
            ls_hx_arrays, unique_hx_val = array_to_hex(ls_renamed_arrays)
            if cluster_json_path is not None:
                hx_to_id_dict = hx_to_id(ls_arrays, ls_hx_arrays)
                mx_path = write_matrixfile_cluster(unique_hx_val, work_path, group_name + '_mx', cluster_map, distances,
                                                   hx_to_id_dict, cluster_score)
            else:
                mx_path = write_matrixfile(unique_hx_val, work_path, group_name + '_mx')

        write_fasta(ls_hx_arrays, ls_names, work_path, group_name)

        run_mafft(os.path.join(work_path, group_name),
                  os.path.join(work_path, group_name), mafft_options, mx_path)

        ls_arrays, ls_names = convert_mafft_output(os.path.join(work_path, group_name + '_output.txt'))

        if pseudo_dna:
            ls_arrays = convert_pseudo_dna_to_spacer_arrays(ls_arrays, dict_unique_id_spacers, dict_alphabets,
                                                            repeat)
            # save_arrays_as_readable_file(group_name, ls_array_names,
            #                              ls_arrays,
            #                              save_path=os.path.join(work_path, 'pseudo_dna'),
            #                              show_gaps=True, easier_spacer_names=False)
        else:
            ls_arrays = reverse_renaming(ls_arrays, dict_reverse_renaming)
            # save_arrays_as_readable_file(group_name, ls_array_names,
            #                              ls_arrays,
            #                              save_path=os.path.join(work_path, 'not_pseudo_dna'),
            #                              show_gaps=True, easier_spacer_names=False)

        write_fasta(ls_arrays, ls_names, work_path, group_name + '_finished')
        crispr_group.update_spacer_arrays_by_ls_arrays_as_list(ls_names, ls_arrays)
        crispr_group.convert_arrays_from_mafft_fmt()

        if cluster_json_path is not None:
            dict_crispr_groups = cluster_renaming(dict_crispr_groups, group_name, cluster_map)

    return dict_crispr_groups


def load_cluster_info(cluster_json_path, group_name):
    with open(cluster_json_path, 'r') as f:
        merged_data = json.load(f)

    data = merged_data[group_name]
    cluster_map = {int(k): v for k, v in data['cluster_map'].items()}

    distances = []
    for cluster_distances in data['distances']:
        int_cluster_distances = {int(k): {int(sub_k): v for sub_k, v in sub_dict.items()} for k, sub_dict in
                                 cluster_distances.items()}
        distances.append(int_cluster_distances)

    return cluster_map, distances


def hx_to_id(ls_arrays, ls_hx_arrays):
    hx_to_id_dict = {}
    for i in range(len(ls_arrays)):
        for j in range(len(ls_arrays[i])):
            hx_to_id_dict[ls_hx_arrays[i][j]] = int(ls_arrays[i][j])
    return hx_to_id_dict


def cluster_renaming(dict_crispr_groups, group_name, cluster_map):
    cluster_to_value_map = {}
    value_map = {}
    new_value = 1

    for spacer_array in dict_crispr_groups[group_name].crispr_dict.values():
        for s in spacer_array.spacer_array:
            if s != '-' and s != '200':
                if cluster_map[int(s)] not in cluster_to_value_map:
                    cluster_to_value_map[cluster_map[int(s)]] = new_value
                    new_value += 1
                value_map[int(s)] = cluster_to_value_map[cluster_map[int(s)]]
        spacer_array.spacer_array = [s if s == '-' else str(value_map[int(s)]) for s in spacer_array.spacer_array]

    return dict_crispr_groups


def compute_reverse_cluster_map(cluster_map):
    reverse_cluster_map = {}
    for key, value in cluster_map.items():
        if value not in reverse_cluster_map:
            reverse_cluster_map[value] = [key]
        else:
            reverse_cluster_map[value].append(key)
    return reverse_cluster_map


def compute_distance_matrices(distances_clust):
    matrices = []
    for item in distances_clust:
        if item:
            unique_indices = set(item.keys())
            for subdict in item.values():
                unique_indices.update(subdict.keys())

            index_to_pos = {index: pos for pos, index in enumerate(sorted(unique_indices))}
            matrix_size = len(unique_indices)

            distances = np.zeros((matrix_size, matrix_size), dtype=int)

            for row, subdict in item.items():
                for col, value in subdict.items():
                    row_pos = index_to_pos[row]
                    col_pos = index_to_pos[col]
                    distances[row_pos, col_pos] = value
                    distances[col_pos, row_pos] = value

            max_distance = distances.max()
            if max_distance > 0:
                scaled_matrix = np.zeros((matrix_size, matrix_size), dtype=int)
                for i in range(len(scaled_matrix)):
                    for j in range(i + 1, len(scaled_matrix)):
                        scaled_dist = np.round(1 + (distances[i][j] / max_distance) * 4).astype(int)
                        scaled_matrix[i][j] = scaled_dist
                        scaled_matrix[j][i] = scaled_dist
            else:
                scaled_matrix = distances

            matrices.append(scaled_matrix)
        else:
            matrices.append(None)

    return matrices


def run_mafft(file_path, output_path, options, mx_path, logger=None):
    file_path_hex2maffttext = os.path.join(MAFFT_PATH, 'hex2maffttext')
    file_path_maffttext2hex = os.path.join(MAFFT_PATH, 'maffttext2hex')
    with open(file_path + '.ascii', 'w') as outfile:
        if logger is not None:
            logger.info('Calling ' + ' '.join([file_path_hex2maffttext] + [file_path + '.txt']))
        else:
            print('Calling ', ' '.join([file_path_hex2maffttext] + [file_path + '.txt']))
        subprocess.call([file_path_hex2maffttext] + [file_path + '.txt'], stdout=outfile)
    with open(output_path + '_output.ascii', 'w') as outfile:
        if logger is not None:
            logger.info('Calling ' + ' '.join(['mafft'] + options + ['--textmatrix', mx_path] + [file_path + '.ascii']))
        else:
            print('Calling ', ' '.join(['mafft'] + options + ['--textmatrix', mx_path] + [file_path + '.ascii']))
        subprocess.call(['mafft'] + options + ['--textmatrix', mx_path] + [file_path + '.ascii'], stdout=outfile)
    with open(output_path + '_output.txt', 'w') as outfile:
        if logger is not None:
            logger.info('Calling ', ' '.join([file_path_maffttext2hex] + [file_path + '_output.ascii']))
        else:
            print('Calling ', ' '.join([file_path_maffttext2hex] + [file_path + '_output.ascii']))
        subprocess.call([file_path_maffttext2hex] + [file_path + '_output.ascii'], stdout=outfile)

    return


def rename_spacers_for_mafft(ls_arrays):
    dict_renaming = {}
    dict_reversed_renaming = {}
    new_name = 1
    renamed_ls_arrays = []
    for i, array in enumerate(ls_arrays):
        new_array = []
        for sp in array:
            if sp not in dict_renaming:
                dict_renaming[sp] = new_name
                dict_reversed_renaming[str(new_name)] = str(sp)
                new_name += 1
            new_array.append(dict_renaming[sp])
        renamed_ls_arrays.append(new_array)
    return renamed_ls_arrays, dict_renaming, dict_reversed_renaming


def reverse_renaming(ls_arrays, dict_reversed_renaming):
    renamed_ls_arrays = []
    for array in ls_arrays:
        new_array = [dict_reversed_renaming.get(sp, '--') for sp in array]
        renamed_ls_arrays.append(new_array)
    return renamed_ls_arrays


def pickle_to_aligned_group(pickle_path, intermediate_save_path, save_path, save_name='0_data.pickle',
                            mafft_options=None):
    too_long_arrays = []
    if mafft_options is None:
        mafft_options = ['--text']
    dict_crispr_by_repeats = import_data.load_crispr_arrays_from_pickle(pickle_path)
    for key, value in dict_crispr_by_repeats.items():
        print(key)
        print(value)
    dict_crispr_by_repeats = raw_data_tools.remove_non_level_4(dict_crispr_by_repeats)
    dict_crispr_aligned = {}

    for repeat, ls_crispr_arrays in dict_crispr_by_repeats.items():
        if ls_crispr_arrays:
            ls_arrays = []
            ls_array_names = ['_'.join([ls_crispr_arrays[i].acc_num, str(i)]) for i in range(len(ls_crispr_arrays))]
            for i, crispr_array in enumerate(ls_crispr_arrays):
                ls_arrays.append(crispr_array.spacer_indexes)
            renamed_ls_arrays, dict_renaming, dict_reversed_renaming = rename_spacers_for_mafft(ls_arrays)
            ls_hex_arrays, unique_hx_values = array_to_hex(renamed_ls_arrays)

            if len(unique_hx_values) > 248:
                too_long_arrays.append((repeat, unique_hx_values))
                print(f'There are more than 248 spacers: {len(unique_hx_values)}')
                continue

            if not os.path.exists(intermediate_save_path):
                os.makedirs(intermediate_save_path)
            mx_path = write_matrixfile(unique_hx_values, intermediate_save_path, repeat + '_mx')
            write_fasta(ls_hex_arrays, ls_array_names, intermediate_save_path, repeat)
            run_mafft(os.path.join(intermediate_save_path, repeat),
                      os.path.join(intermediate_save_path, repeat),
                      mafft_options,
                      mx_path)

            ls_aligned_arrays, ls_aligned_names = convert_mafft_output(
                os.path.join(intermediate_save_path, repeat + '_output.txt'))

            ls_reversed_renaming_aligned_arrays = reverse_renaming(ls_aligned_arrays, dict_reversed_renaming)
            dict_crispr_aligned[repeat] = [
                CRISPRArray(name, crispr.acc_num, crispr.orientation, crispr.evidence_level, crispr.drconsensus,
                            crispr.chromosome_plasmid,
                            crispr.kingdom, crispr.organism_name, array, crispr.cas_type, crispr.cas_gene_info,
                            crispr.cas_gene_sequences) for name, array, crispr in
                zip(ls_aligned_names, ls_reversed_renaming_aligned_arrays, ls_crispr_arrays)]

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(os.path.join(save_path, save_name), 'wb') as handle:
        pickle.dump(dict_crispr_aligned, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # print('Arrays that were too long: ', too_long_arrays)
    print('Number of repeats: ', len(dict_crispr_by_repeats))
    print('Number of aligned repeats: ', len(dict_crispr_aligned),
          'Number of aligned spacer arrays: ', sum([len(val) for val in dict_crispr_aligned.values()]))
    print('Number of repeats with arrays that were too long: ', len(too_long_arrays))
    return dict_crispr_aligned


def pickled_group_to_mafft(path, ls_files, options=None, remove_same_arrays=True):
    if options is None:
        options = DEFAULT_MAFFT_OPTIONS
    data_dict = import_data.read_old_data(path, ls_files)
    ls_arrays, unique_hx_val = array_to_hex(data_dict['arrays'])

    mx_path = write_matrixfile(unique_hx_val, os.path.join('../data', 'pickled_fasta'), '_'.join(ls_files + ['mx']))

    array_names = [a for a in data_dict['metadata']]
    write_fasta(ls_arrays, array_names, os.path.join('../data', 'pickled_fasta'), '_'.join(ls_files))
    run_mafft(os.path.join('../data', 'pickled_fasta', '_'.join(ls_files)),
              os.path.join('../data', 'pickled_fasta', '_'.join(ls_files)), options, mx_path)
    ls_arrays, ls_names = convert_mafft_output(
        os.path.join('../data', 'pickled_fasta', '_'.join(ls_files) + '_output.txt'))
    write_fasta(ls_arrays, ls_names, os.path.join('../data', 'pickled_fasta'), '_'.join(ls_files) + '_finished')
    return ls_arrays, ls_names, data_dict['head']
