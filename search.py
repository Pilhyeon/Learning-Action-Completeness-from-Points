import torch
import utils
import copy
from joblib import Parallel, delayed
from decode import Decoder
from grammar import SingleTranscriptGrammar
import torch.utils.data as data


def optimal_sequence_search(net, config, logger, train_loader):
    with torch.no_grad():
        net.eval()

        cas_lst = [-1] * len(train_loader.dataset)
        act_seed_lst = [-1] * len(train_loader.dataset)
        bkg_seed_lst = [-1] * len(train_loader.dataset)

        temp_loader = data.DataLoader(
            train_loader.dataset,
            batch_size=1,
            shuffle=True, num_workers=config.num_workers)

        loader_iter = iter(temp_loader)
        
        for i in range(len(train_loader.dataset)):
            _index, _data, _label, _point_anno, _, _, _ = next(loader_iter)

            _data = _data.cuda()
            _label = _label.cuda()
            _point_anno = _point_anno.cuda()
            
            _, cas_sigmoid_fuse, _ = net(_data, _label)
                        
            act_seed, bkg_seed = utils.select_seed(cas_sigmoid_fuse.detach().cpu(), _point_anno.detach().cpu())
                        
            cas_lst[_index[0]] = cas_sigmoid_fuse[0].detach().cpu()
            act_seed_lst[_index[0]] = act_seed[0].detach().cpu()
            bkg_seed_lst[_index[0]] = bkg_seed[0].detach().cpu()

        res = Parallel(n_jobs=16)(delayed(greedy_search_with_id)(config.budget, cas, act_seed, bkg_seed) for (cas, act_seed, bkg_seed) in zip(cas_lst, act_seed_lst, bkg_seed_lst))
        
        pseudo_labels, sequence_score = zip(*res)

        train_loader.dataset.stored_info_all['new_dense_anno'] = copy.deepcopy(list(pseudo_labels))
        train_loader.dataset.stored_info_all['sequence_score'] = copy.deepcopy(list(sequence_score))

        return


def greedy_search_with_id(budget, cas_sigmoid_fuse, act_seed, bkg_seed):
    pseudo_labels = torch.zeros_like(act_seed)
    thresh = 0.1
    decoder = Decoder(grammar=None, frame_sampling=1, max_hypotheses=budget, thresh=thresh)
    sequence_scores = []

    for c in range(act_seed.shape[1]):
        num_act_instances = act_seed[:, c].sum()
        act_idx = torch.nonzero(act_seed[:, c], as_tuple=False).squeeze(1)
    
        if num_act_instances < 1:
            continue

        other_actions = act_seed.max(dim=1)[0] - act_seed[:,c]
        other_actions[other_actions < 0] = 0

        bkg_seed_c = bkg_seed + other_actions

        act_seq = [[act_idx[0].item()]]

        if act_idx[0] > 0:
            if bkg_seed_c[:act_idx[0]].sum() > 0:
                max_idx = act_idx[0] - 1

                while bkg_seed_c[max_idx] == 0:
                    max_idx -= 1

                bkg_seed_c[:max_idx + 1] = 1
                
                transcript = [0, 1]
            else:
                act_seq = [[-1, act_idx[0].item()]]
                transcript = [1]
        else:
            transcript = [1]

        prev_loc = act_idx[0]

        for i in range(1, len(act_idx)):
            if (act_idx[i] - prev_loc) > 1:
                transcript += [0, 1]
                act_seq.append([act_idx[i].item()])

                if bkg_seed_c[act_idx[i-1] + 1:act_idx[i]].sum() > 0:
                    min_idx = act_idx[i-1] + 1
                    max_idx = act_idx[i] - 1

                    while bkg_seed_c[min_idx] == 0:
                        min_idx += 1
                    while bkg_seed_c[max_idx] == 0:
                        max_idx -= 1

                    bkg_seed_c[min_idx:max_idx + 1] = 1
            else:
                act_seq[-1].append(act_idx[i].item())
                
            prev_loc = act_idx[i]

        if act_idx[-1] < (act_seed.shape[0] - 1):
            if bkg_seed_c[act_idx[-1] + 1:].sum() > 0:
                min_idx = act_idx[-1] + 1

                while bkg_seed_c[min_idx] == 0:
                    min_idx += 1
                    
                bkg_seed_c[min_idx:] = 1

                transcript += [0]
            else:
                act_seq[-1].append(act_seed.shape[0] - 1)
                                        
        bkg_idx = torch.nonzero(bkg_seed_c, as_tuple=False).squeeze(1)
        bkg_seq = [[bkg_idx[0].item()]]

        prev_loc = bkg_idx[0]

        for i in range(1, len(bkg_idx)):
            if (bkg_idx[i] - prev_loc) > 1:
                bkg_seq.append([bkg_idx[i].item()])
            else:
                bkg_seq[-1].append(bkg_idx[i].item())
                
            prev_loc = bkg_idx[i].clone()
        
        act_seq_new = []
        for item in act_seq:
            act_seq_new.append([item[0], item[-1]])
        bkg_seq_new = []
        for item in bkg_seq:
            bkg_seq_new.append([item[0], item[-1]])

        range_lst = []

        for act_item in act_seq_new:
            x1_act, x2_act = act_item
            min_x = x1_act
            max_x = x2_act
            tmp_del = []
            for bkg_item in bkg_seq_new:
                x1_bkg, x2_bkg = bkg_item

                if x1_bkg < x1_act:
                    if x1_bkg < min_x:
                        min_x = x1_bkg
                    max_x = x2_bkg
                    tmp_del.append(bkg_item)
                else:
                    break
            for item_del in tmp_del:
                bkg_seq_new.remove(item_del)
            if min_x != x1_act:
                range_lst.append([min_x, max_x])
            range_lst.append([x1_act, x2_act])
                
        if len(bkg_seq_new) > 0:
            range_lst.append([bkg_seq_new[0][0], bkg_seq_new[-1][-1]])

        range_lst.append([act_seed.shape[0], act_seed.shape[0]])

        assert len(transcript) == (len(range_lst) - 1)

        decoder.grammar = SingleTranscriptGrammar(transcript, range_lst, n_classes=2)

        act_scores = cas_sigmoid_fuse[:, c].unsqueeze(1)
        bkg_scores = 1 - cas_sigmoid_fuse[:, c].unsqueeze(1)
        scores = torch.cat((bkg_scores, act_scores), dim=1)

        labels, score = decoder.decode(scores.cpu().numpy())

        pseudo_labels[:, c] = torch.tensor(labels)

        sequence_scores.append(score)

    if len(sequence_scores) == 0:
        sequence_score = 0
    else:
        sequence_score = sum(sequence_scores) / len(sequence_scores)

    return pseudo_labels, sequence_score
