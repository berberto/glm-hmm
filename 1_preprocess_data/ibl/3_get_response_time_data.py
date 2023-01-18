# Obtain IBL response time data for producing Figure 6
# Write out the response times and the corresponding sessions
import os

import numpy as np
import numpy.random as npr
from one_global import one

from preprocessing_utils import load_animal_eid_dict, load_data, get_session_id

npr.seed(65)

if __name__ == '__main__':
    ibl_data_path = "../../data/ibl/"
    animal_eid_dict = load_animal_eid_dict(
        ibl_data_path + 'data_for_cluster/final_animal_eid_dict.json')
    # must change directory for working with ONE
    os.chdir(ibl_data_path)
    data_dir = 'response_times/data_by_animal/'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    for animal in animal_eid_dict.keys():
        print(animal)
        animal_inpt, animal_y, animal_session = load_data(
            'data_for_cluster/data_by_animal/' + animal + '_processed.npz')
        for z, eid in enumerate(animal_eid_dict[animal]):

            print(eid)

            trial_data = one.load_object(eid, 'trials', collection='alf')
            choice = trial_data.choice

            session_id = get_session_id(eid, one)
            full_sess_len = len(choice)

            try:
                globals()["feedback_times"] = trial_data.feedback_times
            except AttributeError:
                print(f"Issue saving 'feedback_times' for {eid}")
                globals()['feedback_times'] = np.nan*np.ones((full_sess_len, ))
            try:
                globals()["response_times"] = trial_data.response_times
            except AttributeError:
                print(f"Issue saving 'response_times' for {eid}")
                globals()['response_times'] = np.nan*np.ones((full_sess_len, ))
            try:
                globals()["go_cues"] = trial_data.goCue_times
            except AttributeError:
                print(f"Issue saving 'go_cues' for {eid}")
                globals()['go_cues'] = np.nan*np.ones((full_sess_len, ))
            try:
                globals()["stim_on_times"] = trial_data.stimOn_times
            except AttributeError:
                print(f"Issue saving 'stim_on_times' for {eid}")
                globals()['stim_on_times'] = np.nan*np.ones((full_sess_len, ))

            start = np.nanmin(np.c_[stim_on_times, go_cues], axis=1)

            if (len(feedback_times) == len(response_times)): # some response
                # times/feedback times are missing, so fill these as best as
                # possible
                end = np.nanmin(np.c_[feedback_times, response_times], axis=1)
            elif len(feedback_times) == full_sess_len:
                end = feedback_times
            elif len(response_times) == full_sess_len:
                end = response_times

            # check timestamps increasing:
            idx_to_change = np.where(start > end)[0]

            if len(idx_to_change) > 0:
                start[idx_to_change[0]] = np.nan
                end[idx_to_change[0]] = np.nan

            # Check we have times for at least some trials
            nan_trial = np.isnan(np.c_[start, end]).any(axis=1)

            is_increasing = (((start < end) | nan_trial).all() and
                    ((np.diff(start) > 0) | np.isnan(
                        np.diff(start))).all())

            if is_increasing and ~nan_trial.all() and len(start) == \
                    full_sess_len and len(end) == full_sess_len: #
                # check that times are increasing and that len(start) ==
                # full_sess_len etc
                prob_left_dta = trial_data.probabilityLeft
                assert start.shape[0] == prob_left_dta.shape[0],\
                    "different lengths for prob left and raw response dta: " + \
                    str(start.shape[0]) + " vs " + str(
                        prob_left_dta.shape[0])

                # subset to trials corresponding to prob_left == 0.5:
                unbiased_idx = np.where(prob_left_dta == 0.5)
                response_dta = end[unbiased_idx] - start[unbiased_idx]

                if ((np.nanmedian(response_dta) >= 10) | (np.nanmedian(
                        response_dta) == np.nan)): # check that median
                    # response time for session is less than 10 seconds
                    response_dta = np.array([np.nan for i in range(len(
                        unbiased_idx[0]))])

                rt_sess = [session_id for i in range(response_dta.shape[0])]
                # before saving, confirm that there are as many trials as in
                # some of the other data:
                assert len(rt_sess) == animal_inpt[np.where(animal_session ==
                                                            session_id),
                                       :].shape[1], "response dta is different " \
                                                    "shape compared to inpt"
            else: # if any of the conditions above fail, fill the session's
                # data with nans
                len_prob_50 = animal_inpt[np.where(animal_session ==
                                                            session_id),
                              :].shape[1]
                response_dta = np.array([np.nan for i in range(len_prob_50)])
                rt_sess = [session_id for i in range(response_dta.shape[0])]

            if z == 0:
                rt_session_dta_this_animal = rt_sess
                response_dta_this_animal = response_dta
            else:
                rt_session_dta_this_animal = np.concatenate(
                    (rt_session_dta_this_animal, rt_sess))
                response_dta_this_animal = np.concatenate(
                    (response_dta_this_animal, response_dta))

        assert len(response_dta_this_animal) == len(animal_inpt), "different size for response times and inpt"
        np.savez(data_dir + animal + '.npz', response_dta_this_animal,
                 rt_session_dta_this_animal)
