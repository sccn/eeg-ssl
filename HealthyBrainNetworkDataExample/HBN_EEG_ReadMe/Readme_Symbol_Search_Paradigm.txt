Filenames:

xxx_WISC_ProcSpeed = Symbol Search Paradigm

Triggers are as following:
92 = Start of Symbol Search Paradigm;
20 = Start of new page
14 = response for trial



trial_duration_x:  reaction time for each trial (better to extract reaction time based on markers)
CD_START: marker for starting the paradigm
CD_Buttons: marker buttons (actually only 14 was used as button press)
CD_NXTPG: marker for new page of trials
activated_respx: button response of subject. associated with correct_resp_allpages; First 2 = end of the experiment; par.activated_resp
correct_resp_allpages: 0 = target not in probes, correct response is NO; 1 = target in probes, correct response is YES
useEL: eye-tracker was recorded if 1
useEL_Calib: eye-tracker calibration was performed if 1
recordEEG: EEG was recorded if 1

The correct answers for the responses are saved in:

Symbol_Search_correct_response.mat or Symbol_Search_correct_response.xls
