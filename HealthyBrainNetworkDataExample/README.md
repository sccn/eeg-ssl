## HBN_EEG_ReadMe
- `Readme_General.txt`: Describe general filenaming pattern and event trigger 
- `Readme_<filename>.txt`: 
	- First half: Describe task specific trigger codes 
	- Second half: Describe fields in the `par` variable of the task behavioral .mat file
## NDARZZ993CEV
Behavioral, EEG, and eyetracking data are organized in subject directories, identified by subject ID (e.g. NDARZZ993CEV).


- `EEG/raw/mat_format`: store raw EEG data and event triggers in [EEGLAB format](https://eeglab.org/tutorials/ConceptsGuide/Data_Structures.html#eeglab-data-structures).
- `EEG/raw/csv_format`: store raw data in `<taskname>_data.csv` and event triggers in `<taskname>_event.csv` fiels.

## ds004186
 HBN resting state data in BIDS format


## Some BIDS documentation
- BIDS specification: https://bids-specification.readthedocs.io/en/stable/modality-agnostic-files.html
documentation

- BIDS schema: 
  - Description: https://bids-specification.readthedocs.io/en/stable/appendices/schema.html
  - Repo: https://github.com/bids-standard/bids-specification/tree/master/src/schema