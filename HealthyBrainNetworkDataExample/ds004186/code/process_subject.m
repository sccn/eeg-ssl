% Process a single HBN subject for BIDS export
%
% Arnaud Delorme and Dung Truong, 2022

function process_subject(inpath, folderout, subj, gender, age, handedness)
    addpath('/expanse/projects/nemar/eeglab');
    eeglab nogui; close;

    fileName = fullfile(inpath, subj, 'EEG/raw/mat_format/RestingState.mat');
    %infoRow = strmatch(subj, info(:,1)', 'exact');
    if exist(fileName, 'file') % && length(infoRow) > 0
        subjoutfolder = fullfile(folderout, subj);
        if ~exist(subjoutfolder,'dir')
            mkdir(subjoutfolder);
        end
        EEG = load(fileName);
        EEG = EEG.EEG;

        EEG
        [events,counts] = eeg_eventtypes(EEG)

        % Write events to a file
        fid2 = fopen(fullfile(folderout, 'jobs', [subj '_events.txt' ] ), 'w');
        for iEvent = 1:length(events)
            fprintf(fid2, '%s, %s -> %d; ', subj, events{iEvent}, counts(iEvent));
        end
        fprintf(fid2, '\n');
        fclose(fid2);

        if EEG.nbchan == 129
            fprintf('Processing subject %s\n', subj);
            for iEvent2 = 1:length(EEG.event)
                EEG.event(iEvent2).latency = EEG.event(iEvent2).sample;
            end
            
            % copy info
            EEG.gender     = gender; %info{infoRow(1),2};
            EEG.age        = age; %info{infoRow(1),3};
            EEG.handedness = handedness; %info{infoRow(1),4};
            EEG.subject     = subj;
            EEG = pop_chanedit(EEG, 'load',{'GSN_HydroCel_129.sfp','filetype','autodetect'});

            % resting state: instruction. Open 20 close 40 20 - 40
            fprintf('\tProcessing eyesopen\n');
            EEGeyeso = pop_epoch( EEG, {  '20  ' }, [0  19.995], 'newname', 'Eyes open', 'epochinfo', 'yes');
            if EEGeyeso.trials > 5
                EEGeyeso = pop_select(EEGeyeso, 'notrial', 6);
                EEGeyeso.etc.notes = 'Long recording; forgot to stop recording';
            end
            % EEGeyeso.event = [];
            for irun=1:numel(EEGeyeso.epoch)
                fprintf('\t\tProcessing run %d\n', irun);
                fileNameOpenSet   = fullfile(subjoutfolder, [ subj '_task-EO_run-0' num2str(irun) '_eeg.set' ]);
                %if exist(fileNameOpenSet, 'file')
                    %disp([fileNameOpenSet ' exists. Skipped']);
                    %continue;
                %end
                
                EEGeyeso_run = EEGeyeso;
                EEGeyeso_run.data = EEGeyeso_run.data(:,:,irun);
                EEGeyeso_run.epoch = [];
                EEGeyeso_run.event = EEGeyeso_run.event(irun); % re-added
                EEGeyeso_run.run = irun;
                EEGeyeso_run = eeg_checkset( EEGeyeso_run );

                fid = fopen([fileNameOpenSet(1:end-8) '_dataqual.txt'],'w');
                fprintf(fid,'Event count:\n');
                [types, count] = eeg_eventtypes(EEGeyeso_run);
                for t=1:numel(types)
                    fprintf(fid,'\t%s\t%d\n', types{t}, count(t));
                end
                EEGeyeso_run.event = [];

                pop_saveset(EEGeyeso_run, 'filename', fileNameOpenSet, 'savemode', 'onefile');

                EEGTemp = pop_clean_rawdata(EEGeyeso_run, 'FlatlineCriterion',5,'ChannelCriterion',0.7,'LineNoiseCriterion',4,'Highpass','off','BurstCriterion','off','WindowCriterion','off','BurstRejection','off','Distance','Euclidian');
                norichans = EEGeyeso_run.nbchans;
                ngoodchans = numel(EEGTemp.chanlocs);
                ngooddata = size(EEGTemp.data,2);
                noridata = size(EEGeyeso_run.data,2);

                fprintf(fid,'Number of removed channels: %d\n', norichans-ngoodchans);
                fprintf(fid,'Percent of removed data: %.2f',100*(noridata-ngooddata)/noridata);
                fclose(fid);
            end

            fprintf('\tProcessing eyesclosed\n');
            EEGeyesc = pop_epoch( EEG, {  '30  ' }, [0  39.995], 'newname', 'Eyes closed', 'epochinfo', 'yes');
            %EEGeyesc.event = [];
            for irun=1:numel(EEGeyesc.epoch)
                fprintf('\t\tProcessing run %d\n', irun);
                fileNameClosedSet = fullfile(subjoutfolder, [ subj '_task-EC_run-0' num2str(irun) '_eeg.set' ]);
                %if exist(fileNameClosedSet, 'file')
                    %disp([fileNameClosedSet ' exists. Skipped']);
                    %continue;
                %end
                EEGeyesc_run = EEGeyesc;
                EEGeyesc_run.data = EEGeyesc_run.data(:,:,irun);
                EEGeyesc_run.epoch = [];
                EEGeyesc_run.event = EEGeyesc_run.event(irun);
                EEGeyesc_run.run = irun;
                EEGeyesc_run = eeg_checkset( EEGeyesc_run );

                fid = fopen([fileNameClosedSet(1:end-8) '_dataqual.txt'],'w');
                fprintf(fid,'Event count:\n');
                [types, count] = eeg_eventtypes(EEGeyeso_run);
                numel(types)
                for t=1:numel(types)
                    fprintf(fid,'\t%s\t%d\n', types{t}, count(t));
                end
                EEGeyesc_run.event = [];

                pop_saveset(EEGeyesc_run, 'filename', fileNameClosedSet, 'savemode', 'onefile');

                EEGTemp = pop_clean_rawdata(EEGeyesc_run, 'FlatlineCriterion',5,'ChannelCriterion',0.7,'LineNoiseCriterion',4,'Highpass','off','BurstCriterion','off','WindowCriterion','off','BurstRejection','off','Distance','Euclidian');
                norichans = EEGeyesc_run.nbchans;
                ngoodchans = numel(EEGTemp.chanlocs);
                ngooddata = size(EEGTemp.data,2);
                noridata = size(EEGeyesc_run.data,2);
                fprintf(fid,'Number of removed channels: %d\n', norichans-ngoodchans);
                fprintf(fid,'Percent of removed data: %.2f',100*(noridata-ngooddata)/noridata);
                fclose(fid);
            end
        else
            error('Not 129 channels');
        end
    end
end
