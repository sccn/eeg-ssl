% Same as example 3 but uses different sessions and runs
% This script assumes you have used process_subject on all subjects
% 
% Arnaud Delorme - May 2022

% list the releases to export
releases = { 'release1.1' }

allPheno = readtable('HBN_All_Pheno.csv');
allStatus = readtable('subject_status.csv');

data = [];
pInfo = { 'participant_id' 'gender' 'age' 'handedness' 'commercial_use' 'full_pheno' };

for iFolder = 1:length(releases)
    files = dir(fullfile('/expanse/projects/nemar/child-mind-R21/', releases{iFolder}, 'N*'));

    for iFile = 1:length(files)
        s = files(iFile).name;
        ind1 = strmatch(s, table2cell(allPheno(:,1)), 'exact');
        ind2 = strmatch(s, table2cell(allStatus(:,1)), 'exact');

        if isempty(ind1)
            error('Subject %s/%s not found in pheno file', releases{iFolder}, files(iFile).name);
        end
        if isempty(ind2)
            error('Subject %s/%s not found in status file', releases{iFolder}, files(iFile).name);
        end    

        if allStatus{ind2,3}
            pInfo =  [ pInfo; table2cell(allPheno(ind1(1),:)) ];
            data(end+1).file = { ...
            fullfile(s,  [ s '_task-EC_run-01_eeg.set' ] ) ...
            fullfile(s,  [ s '_task-EC_run-02_eeg.set' ] ) ...
            fullfile(s,  [ s '_task-EC_run-03_eeg.set' ] ) ...
            fullfile(s,  [ s '_task-EC_run-04_eeg.set' ] ) ...
            fullfile(s,  [ s '_task-EC_run-05_eeg.set' ] ) ...
            fullfile(s,  [ s '_task-EO_run-01_eeg.set' ] ) ...
            fullfile(s,  [ s '_task-EO_run-02_eeg.set' ] ) ...
            fullfile(s,  [ s '_task-EO_run-03_eeg.set' ] ) ...
            fullfile(s,  [ s '_task-EO_run-04_eeg.set' ] ) ...
            fullfile(s,  [ s '_task-EO_run-05_eeg.set' ] ) };
            data(end  ).run     = [   1    2    3    4    5    1    2    3    4    5   ];
            data(end  ).task    = { 'EC' 'EC' 'EC' 'EC' 'EC' 'EO' 'EO' 'EO' 'EO' 'EO' };
        else
            fprintf('Subject %s skipped\n', files(iFile).name);
        end
    end
end

%% general information for dataset_description.json file
% -----------------------------------------------------
generalInfo.Name = 'HBN EO/EC task';
generalInfo.ReferencesAndLinks = { 'No bibliographic reference other than the DOI for this dataset' };
generalInfo.BIDSVersion = 'v1.2.1';
generalInfo.License = 'CC0';
generalInfo.Authors = { 'Michael Milham' 'Arnaud Delorme' 'Dung Truong' };

%% participant column description for participants.json file
% ---------------------------------------------------------
pInfoDesc.participant_id.LongName    = 'Participant identifier';
pInfoDesc.participant_id.Description = 'Unique participant identifier';

pInfoDesc.gender.Description = 'Sex of the participant';
pInfoDesc.gender.Levels.M    = 'male';
pInfoDesc.gender.Levels.F    = 'female';

pInfoDesc.age.Description = 'age of the participant';
pInfoDesc.age.Units       = 'years';

%% Content for README file
% -----------------------
README = [ 'HBN EO/EC datasets                   ' 10 ...
'The Healthy Brain Network recorded a single eyes-open and eyes-closed EEG recording with events for switching between states. Eyes-open lasted for 20 second and eyes-closed for 40 seconds. Because such files are difficult to process, they have been segumented here with 5 runs for EO and 5 runs for EC. See notes for included datasets.' ];

%% Content for CHANGES file
% ------------------------
CHANGES = sprintf([ 'Version 1.0 - XXX 2022\n' ...
                    ' - Initial release\n' ]);

%% Task information for xxxx-eeg.json file
% ---------------------------------------
tInfo.InstitutionAddress = '9500 Gilman Drive, La Jolla CA 92093, USA';
tInfo.InstitutionName = 'University of California, San Diego';
tInfo.InstitutionalDepartmentName = 'Institute of Neural Computation';
tInfo.PowerLineFrequency = 60;
tInfo.ManufacturersModelName = 'EGI';
%tInfo.Reference = 'Delorme A, Westerfield M, Makeig S. Medial prefrontal theta bursts precede rapid motor responses during visual selective attention. J Neurosci. 2007 Oct 31;27(44):11949-59. doi: 10.1523/JNEUROSCI.3477-07.2007. PMID: 17978035; PMCID: PMC6673364.'
% tInfo.Instructions


% call to the export function
% ---------------------------
targetFolder =  './HBN-eo-ec';
bids_export(data, ...
    'targetdir', targetFolder, ...
    'taskName', 'P300',...
    'gInfo', generalInfo, ...
    'pInfo', pInfo, ...
    'pInfoDesc', pInfoDesc, ...
    'README', README, ...
    'CHANGES', CHANGES, ...
    'renametype', {}, ...
    'tInfo', tInfo, ...
    'copydata', 1);
 
fprintf(2, 'WHAT TO DO NEXT?')
fprintf(2, ' -> upload the %s folder to http://openneuro.org to check it is valid\n', targetFolder);

