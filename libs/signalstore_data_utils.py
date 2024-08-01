from pathlib import Path
import scipy.io as sio
import numpy as np
import xarray as xr
import os
from os import scandir, walk
from signalstore import UnitOfWorkProvider
# from mongomock import MongoClient
from pymongo.mongo_client import MongoClient
from fsspec.implementations.local import LocalFileSystem
from fsspec import get_mapper
from fsspec.implementations.dirfs import DirFileSystem
import fsspec
import mne
import pandas as pd
import json
# from dask.distributed import LocalCluster


class SignalstoreHBN():
    def __init__(self, 
                 data_path='/mnt/nemar/openneuro/ds004186',                                     # path to raw data
                 cache_path='/mnt/nemar/dtyoung/eeg-ssl-data/signalstore/hbn',                  # path where signalstore netCDF files are stored
                 dataset_name="healthy_brain_network",                                          # TODO right now this is resting state data --> rename it to differentiate between tasks later
                 ):
        filesystem = LocalFileSystem()
        # tmp_dir = TemporaryDirectory()
        # print(tmp_dir.name)

        # Create data storage location
        self.dataset_name = dataset_name
        self.data_path = Path(data_path)

        # Create a directory for the dataset
        store_path = Path(cache_path)
        if not os.path.exists(store_path):
            os.makedirs(store_path)

        tmp_dir_fs = DirFileSystem(
            store_path,
            filesystem=filesystem
        )
        uri = "mongodb+srv://dtyoung112:XbiUEbzmCacjafGu@cluster0.6jtigmc.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
        # Create a new client and connect to the server
        client = MongoClient(uri)
        memory_store = {}
        self.uow_provider = UnitOfWorkProvider(
            mongo_client=client,
            filesystem=tmp_dir_fs,
            memory_store=memory_store
        )

        self.uow = self.uow_provider(self.dataset_name)
        # self.load_domain_models()
        # self.add_data()

    def load_domain_models(self):
        cwd = Path.cwd()
        domain_models_path = cwd.parent / f"DomainModels/{self.dataset_name}/data_models.json"
        metamodel_path = cwd.parent / f"DomainModels/{self.dataset_name}/metamodels.json"
        property_path = cwd.parent / f"DomainModels/{self.dataset_name}/property_models.json"
        with open(metamodel_path) as f:
            metamodels = json.load(f)

        with open(property_path) as f:
            property_models = json.load(f)
            
        # load domain models json file
        with open(domain_models_path) as f:
            domain_models = json.load(f)
            
        with self.uow as uow:
            for property_model in property_models:
                uow.domain_models.add(property_model)
                model = uow.domain_models.get(property_model['schema_name'])
                print('property model: ', model['schema_name'])
            for metamodel in metamodels:
                uow.domain_models.add(metamodel)
                model = uow.domain_models.get(metamodel['schema_name'])
                print('meta model: ', model['schema_name'])
            for domain_model in domain_models:
                uow.domain_models.add(domain_model)
                model = uow.domain_models.get(domain_model['schema_name'])
                print('domain model: ', model['schema_name'])
                uow.commit()

    def load_eeg_data_from_bids(sefl, bids_data_path):
        for entry in scandir(bids_data_path):
            if entry.is_dir() and entry.name.startswith('sub-'):
                subject_dir = entry.name
                subject = subject_dir.split('-')[1]
                subject_dir_path = bids_data_path / subject_dir
                eeg_dir = subject_dir_path / "eeg"

                tasks = ['EC', 'EO']
                runs  = [list(range(1, 6)), list(range(1, 6))]
                for t, task in enumerate(tasks):
                    for run in runs[t]:
                        # get file by name pattern subject_dir*task*run_eeg.set
                        raw_file = eeg_dir / f"{subject_dir}_task-{task}_run-{run}_eeg.set"
                        print('raw file', raw_file)
                        if not os.path.exists(raw_file):
                            continue

                        EEG = mne.io.read_raw_eeglab(os.path.join(raw_file), preload=True)
                        eeg_data = EEG.get_data()

                        print('data shape:', eeg_data.shape)
                        
                        eeg_json_file = eeg_dir / f"{subject_dir}_task-{task}_run-{run}_eeg.json"
                        eeg_json = json.load(eeg_json_file.open())
                        fs = int(eeg_json['SamplingFrequency'])
                        max_time = eeg_data.shape[1] / fs
                        time_steps = np.linspace(0, max_time, eeg_data.shape[1]).squeeze() # in seconds
                        print('time steps', len(time_steps))

                        channel_coords_file = eeg_dir / f"{subject_dir}_task-{task}_run-{run}_channels.tsv"
                        channel_coords = pd.read_csv(channel_coords_file, sep='\t') 
                        print('channel coords file len', len(channel_coords))
                        # get channel names from channel_coords
                        channel_names = channel_coords['name'].values
                        print('channel coords names', channel_names)
                        print(len(channel_names))
                        eeg_xarray = xr.DataArray(
                            data=eeg_data,
                            dims=['channel','time'],
                            coords={
                                'time': time_steps,
                                'channel': channel_names
                            },
                            attrs={
                                'schema_ref': 'eeg_signal',
                                'data_name': f"{subject_dir}_task-{task}_run-{run}",
                                'subject': f'{subject}',
                                'modality': 'EEG',
                                'task': task,
                                'session_run': run,
                                'sampling_frequency': fs,
                            }
                        )
                        yield eeg_xarray

    def add_data(self):
        for eeg_xarray in self.load_eeg_data_from_bids(self.data_path):
            with self.uow_provider(self.dataset_name) as uow:
                query = {
                    "schema_ref": "eeg_signal",
                    "data_name": eeg_xarray.attrs['data_name']
                }
                sessions = uow.data.find(query)
                if len(sessions) == 0:
                    print('adding data', eeg_xarray.attrs['data_name'])
                    uow.data.add(eeg_xarray)
                    uow.commit()

    def remove_all(self):
        with self.uow_provider(self.dataset_name) as uow:
            sessions = uow.data.find({})
            print(len(sessions))
            for session in range(len(sessions)):
                uow.data.remove(session['schema_ref'], session['data_name'])
                uow.commit()

            uow.purge()
            
            print('Verifying deletion job. Dataset length: ', len(uow.data.find({})))

    def query_data(self, query={}):
        with self.uow_provider(self.dataset_name) as uow:
            sessions = uow.data.find(query)
            print(f'Found {len(sessions)} records')
            
            return sessions


if __name__ == "__main__":
    sstore_hbn = SignalstoreHBN()
    sstore_hbn.add_data()