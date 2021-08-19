import numpy as np


class Decoder(object):

    class TracebackNode(object):
        def __init__(self, label, predecessor, boundary = False):
            self.label = label
            self.predecessor = predecessor
            self.boundary = boundary

    class HypDict(dict):
        class Hypothesis(object):
            def __init__(self, score, cur_len, traceback):
                self.score = score
                self.cur_len = cur_len
                self.traceback = traceback
        def update(self, key, score, cur_len, traceback):
            if (not key in self) or (self[key].score <= score):
                self[key] = self.Hypothesis(score, cur_len, traceback)

    def __init__(self, grammar, frame_sampling=1, max_hypotheses=np.inf, thresh=0.1):
        self.grammar = grammar
        self.frame_sampling = frame_sampling
        self.max_hypotheses = max_hypotheses
        self.thresh = thresh

    def decode(self, log_frame_probs):
        frame_scores = log_frame_probs
        # create initial hypotheses
        hyps = self.init_decoding(frame_scores)
        # decode each following time step
        for t in range(2 * self.frame_sampling - 1, frame_scores.shape[0], self.frame_sampling):
            hyps = self.decode_frame(t, hyps, frame_scores)
            self.prune(hyps, self.thresh)
        # transition to end symbol
        final_hyp = self.finalize_decoding(hyps, frame_scores)
        labels, segments = self.traceback(final_hyp, frame_scores.shape[0])
        return labels, final_hyp.score

    ### helper functions ###
    def frame_score(self, frame_scores, t, label):
        return frame_scores[t, label]

    def prune(self, hyps, thresh=None):
        tmp = sorted( [ (hyps[key].score, key) for key in hyps ] )

        half_thresh = thresh

        count = 0
        for i in range(len(tmp) - 1):
            if tmp[i][0] < half_thresh:
                count += 1
            else:
                break

        del_keys = [ x[1] for x in tmp[0 : count] ]
        for key in del_keys:
            del hyps[key]

        if len(hyps) > self.max_hypotheses:
            tmp = sorted( [ (hyps[key].score, key) for key in hyps ] )

            end = len(tmp) - self.max_hypotheses
            base_score = tmp[end][0]

            i = 0
            for i in range(end-1, -1, -1):
                if base_score == tmp[i][0]:
                    end -= 1
                else:
                    break

            del_keys = [ x[1] for x in tmp[0 : end] ]
            for key in del_keys:
                del hyps[key]

    def init_decoding(self, frame_scores):
        hyps = self.HypDict()
        context = (self.grammar.start_symbol(),)
        for label in self.grammar.possible_successors(context):
            key = context + (label, self.frame_sampling, '0')            
            score = np.inf
            hyps.update(key, score, 1, self.TracebackNode(label, None, boundary = True))
        return hyps

    def decode_frame(self, t, old_hyp, frame_scores):
        new_hyp = self.HypDict()
        for key, hyp in old_hyp.items():
            context, label, length, hist = key[0:-3], key[-3], key[-2], key[-1]

            full_context = tuple(list(context) + [label])
            cur_idx_1, cur_idx_2 = self.grammar.next_idx_range[context]
            next_idx_1, next_idx_2 = self.grammar.next_idx_range[full_context]

            if hyp.cur_len == next_idx_1:
                new_label = list(self.grammar.possible_successors(full_context))[0]
                if new_label == self.grammar.end_symbol():
                    continue
                hist_sum = hist.split(',')
                hist_sum = [int(item) for item in hist_sum]
                hist_sum = sum(hist_sum)
                new_hist = hist + ',' + str(hyp.cur_len - hist_sum)
                new_key = full_context + (new_label, self.frame_sampling, new_hist)
                score = self.cal_hist_score(new_hist, hyp.cur_len + 1, frame_scores, hyp.score)
                new_hyp.update(new_key, score, hyp.cur_len + 1, self.TracebackNode(new_label, hyp.traceback, boundary = True))
                continue
            
            if hyp.cur_len <= cur_idx_2:
                new_key = context + (label, length + self.frame_sampling, hist)
                score = hyp.score
                new_hyp.update(new_key, score, hyp.cur_len + 1, self.TracebackNode(label, hyp.traceback, boundary = False))
                continue

            # stay in the same label...
            new_key = context + (label, length + self.frame_sampling, hist)
            score = hyp.score
            new_hyp.update(new_key, score, hyp.cur_len + 1, self.TracebackNode(label, hyp.traceback, boundary = False))
            # ... or go to the next label
            new_label = list(self.grammar.possible_successors(full_context))[0]
            if new_label == self.grammar.end_symbol():
                continue
            hist_sum = hist.split(',')
            hist_sum = [int(item) for item in hist_sum]
            hist_sum = sum(hist_sum)
            new_hist = hist + ',' + str(hyp.cur_len - hist_sum)
            new_key = full_context + (new_label, self.frame_sampling, new_hist)
            
            score = self.cal_hist_score(new_hist, hyp.cur_len + 1, frame_scores, hyp.score)
            new_hyp.update(new_key, score, hyp.cur_len + 1, self.TracebackNode(new_label, hyp.traceback, boundary = True))

        return new_hyp

    def finalize_decoding(self, old_hyp, frame_scores):
        final_hyp = self.HypDict.Hypothesis(-np.inf, 0, None)
        for key, hyp in old_hyp.items():
            context, label, length, hist = key[0:-3], key[-3], key[-2], key[-1]
            context = context + (label,)

            labels, segments = self.traceback(hyp, frame_scores.shape[0])
            
            labels = np.array(labels)

            seg_list = np.where(labels > 0)
            seg_list_bkg = np.where(labels < 1)

            score_act = self.oic_score(seg_list, frame_scores[:, 1])
            score_bkg = self.oic_score(seg_list_bkg, frame_scores[:, 0])

            score = sum(score_act + score_bkg) / (len(score_act) + len(score_bkg))

            if score >= final_hyp.score:
                final_hyp.score, final_hyp.traceback = score, hyp.traceback

        return final_hyp

    def traceback(self, hyp, n_frames):
        class Segment(object):
            def __init__(self, label):
                self.label, self.length = label, 0
        traceback = hyp.traceback
        if traceback is None:
            print("", end="")
        labels = []
        segments = [Segment(traceback.label)]
        while not traceback == None:
            segments[-1].length += self.frame_sampling
            labels += [traceback.label] * self.frame_sampling
            if traceback.boundary and not traceback.predecessor == None:
                segments.append( Segment(traceback.predecessor.label) )
            traceback = traceback.predecessor
        segments[0].length += n_frames - len(labels) # append length of missing frames
        labels += [hyp.traceback.label] * (n_frames - len(labels)) # append labels for missing frames
        return list(reversed(labels)), list(reversed(segments))

    def cal_hist_score(self, hist, cur_len, frame_scores, cur_score, _lambda=0.25):
        durations = hist.split(',')

        segments = []

        len_seg = int(durations[-1])

        start = cur_len - len_seg - 1
        end = cur_len - 2
        label = self.grammar.transcript[len(durations) - 2]

        inner_score = np.mean(frame_scores[start:end + 1, label])
        
        outer_s = max(0, int(start - _lambda * len_seg))
        outer_e = min(int(frame_scores.shape[0] - 1), int(end + _lambda * len_seg))

        outer_seg = list(range(outer_s, start)) + list(range(end + 1, outer_e + 1))
        
        if len(outer_seg) == 0:
            outer_score = 0
        else:
            outer_score = np.mean(frame_scores[outer_seg, label])

        if len(durations) == 2:
            score = inner_score - outer_score
        else:
            score = ((cur_score * (len(durations) - 2)) + inner_score - outer_score) / (len(durations) - 1)

        return score

    def oic_score(self, seg_list, frame_scores, _lambda=0.25):
        final_score = []
        temp_list = np.array(seg_list)[0]
        if temp_list.shape[0] > 0:
            grouped_temp_list = grouping(temp_list)
            for j in range(len(grouped_temp_list)):
                inner_score = np.mean(frame_scores[grouped_temp_list[j]])

                len_proposal = len(grouped_temp_list[j])
                outer_s = max(0, int(grouped_temp_list[j][0] - _lambda * len_proposal))
                outer_e = min(int(frame_scores.shape[0] - 1), int(grouped_temp_list[j][-1] + _lambda * len_proposal))

                outer_temp_list = list(range(outer_s, int(grouped_temp_list[j][0]))) + list(range(int(grouped_temp_list[j][-1] + 1), outer_e + 1))
                
                if len(outer_temp_list) == 0:
                    outer_score = 0
                else:
                    outer_score = np.mean(frame_scores[outer_temp_list])

                c_score = inner_score - outer_score
                final_score.append(c_score)

        return final_score

def grouping(arr):
    return np.split(arr, np.where(np.diff(arr) != 1)[0] + 1)
