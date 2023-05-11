addpath('/expanse/projects/nemar/dtyoung/NEMAR-pipeline/eeglab');
eeglab nogui;

dspath = '/expanse/projects/nemar/openneuro/processed/ds004362';
savepath = '/expanse/projects/nemar/dtyoung/eeg-ssl/ds004362';

contents = dir(dspath);
while ~isempty(contents)
	content = contents(end);
	contents = contents(1:end-1);
	if content.isdir == 0 && endsWith(content.name, '.set')
		EEG = pop_loadset(fullfile(content.folder, content.name));
		data = EEG.data;
		nameparts = split(content.name, '.');
		save(fullfile(savepath,nameparts{1}), 'data');
	end
	if content.isdir && ~strcmp(content.name, '.') && ~strcmp(content.name, '..')
		contents = [contents; dir(fullfile(content.folder, content.name))];
	end	
end
