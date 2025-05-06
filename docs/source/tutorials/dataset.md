Build Your Own Dataset
=======================

In high energy physics, commonly, data is stored in a `*.root` file. In this tutorial, you will learn the dataset structure of HyPER and we will show you how to build your own `GraphDataset` from `*.root` files. We are going to use semi-leptonic ttbar events as an example for this tutorial.

Dataset Structure
-----------

HyPER uses `HDF5` to store data, it contains two data groups: `INPUTS` and `LABELS`. The structure of our dataset is as follows:
``` yaml
|- INPUTS           |- LABELS
|--- JET            |--- JET
|--- LEPTON         |--- LEPTON
|--- MET            |--- MET
|--- GLOBAL
```
The `INPUTS` datasets `JET`, `LEPTON` and `MET` have same size along `dim=1`, and contain the "features" of these objects.
The `GLOBAL` dataset stores event-wise information.

Each final state (here `MET` is used as an approximation for the neutrino) is assigned an integer, stored in the corresponding dataset in `LABELS`. This integer can be used in the dataset config file to determine the (hyper)edges to reconstruct.

!!! Note

    Some features are only available for certain objects, such as "charge" for leptons (not jets), "btag" for jets (not leptons). We recommend the use of `0` as a placeholder for the missing features in such case.


Each `HDF5` dataset has a corresponding configuration file `config.yaml`, placed on the directory above the dataset, which tells HyPER how to interpretate the data:
```yaml
input:
  padding: 24 #20 jets + 3 leptons + 1 MET
  nodes: # of form {Node type: enum}. Lists each object as defined in the "INPUTS" group of the h5 files, with an enum which the user can define
    JET: 1
    LEPTON: 2
    MET: 3
  node_features: # Ordered list of node features
    - e
    - eta
    - phi 
    - pt 
    - btag
    - charge
    - id
  node_transforms: # Corresponding list of transformations to the node features
    - torch.log(x)
    - x
    - x
    - torch.log(x)
    - x
    - x
    - x
  edge_features: # List of edge features from a pre-defined list of edge features which HyPER can construct
    - delta_eta
    - delta_phi
    - delta_R
    - M2
  global_features: # Global feature list
    - njet
    - nbJet_90
    - nbJet_85
    - nbJet_77
    - nbJet_70
    - nbJet_65
  global_transforms: # Corresponding list of transformations to the global data
    - x/6
    - x/2
    - x/2
    - x/2
    - x/2
    - x/2
  pre_transform: True # Boolean whether to transform the features before saving the dataset PyG object to file

target: 
# Nodes are of the form [particle_enum - matching index]. particle_enum corresponds to the values of the nodes field above. 
  edge:
    w1: ['2-1','3-1'] #Leptonic W
    w2: ['1-3','1-4'] #Hadronic W
  hyperedge: 
    t1: ['1-1','2-1','3-1'] #Leptonic top
    t2: ['1-2','1-3','1-4'] #Hadronic top
```

The structure of the directories should be the following:
``` yaml
|- config.yaml          
|- raw
|--- yourDataset.h5            
|- processed     
```
where the `processed` directory will contain the tensor prebuilt before the training/evaluation and the `raw` directory contains all your `HDF5` datasets.

Example Script
-----------

The following is the core function of a script used to build a semi-leptonic ttbar dataset.

``` python
def MakeDatasetTtbarSingleLepton(input: str, output: str, split: bool = False, saveEven: bool = True):
    r"""Function to construct HyPER dataset.

    Args:
        input (str): input ROOT file.
        output (str): output HyPER dataset in .h5 format.
        split (bool): save only events with odd/even event number.
        saveEven (bool): if True, will save events with even event number
    """

    # Max number of objects (padding)
    pad_to_jet = 20
    pad_to_lepton = 3

    # Load the ROOT file
    root_file = uproot.open(input)
    tree = root_file['reco']
    num_entries = tree.num_entries

    # ---------------------------------------------------
    #                 Load data from ROOT
    # ---------------------------------------------------
    # Jets features
    jet_bTag  = tree["Jet_GN2v01_PCBT_quantile_NOSYS"].array(library='np')
    jet_eta   = tree["Jet_eta"].array(library='np')
    jet_phi   = tree["Jet_phi"].array(library='np')
    jet_e     = tree["Jet_e"].array(library='np')
    jet_pt    = tree["Jet_pt"].array(library='np')

    # Leptons features
    el_eta   = tree["Electron_eta"].array(library='np')
    el_phi   = tree["Electron_phi"].array(library='np')
    el_e     = tree["Electron_e"].array(library='np')
    el_pt    = tree["Electron_pt"].array(library='np')
    el_q     = tree["Electron_q"].array(library='np')

    mu_eta   = tree["Muon_eta"].array(library='np')
    mu_phi   = tree["Muon_phi"].array(library='np')
    mu_e     = tree["Muon_e"].array(library='np')
    mu_pt    = tree["Muon_pt"].array(library='np')
    mu_q     = tree["Muon_q"].array(library='np')

    lep_eta = [np.concatenate([el, mu], axis=0) for el, mu in zip(el_eta, mu_eta)]
    lep_phi = [np.concatenate([el, mu], axis=0) for el, mu in zip(el_phi, mu_phi)]
    lep_e   = [np.concatenate([el, mu], axis=0) for el, mu in zip(el_e, mu_e)]
    lep_pt  = [np.concatenate([el, mu], axis=0) for el, mu in zip(el_pt, mu_pt)]
    lep_q   = [np.concatenate([el, mu], axis=0) for el, mu in zip(el_q, mu_q)]

    lep_id = np.full((num_entries, pad_to_lepton), 3) #id is 3 for muons, 2 for electrons
    el_pt_forCount = [] #used only to count the number of electrons per event
    for arr in el_pt: #fill empty with Nan to use isnan
        if arr.size == 0:
            el_pt_forCount.append([np.nan]) 
        else:
            el_pt_forCount.append(arr)
    el_pt_forCount = np.array(el_pt_forCount, dtype=np.float32)
    num_electrons_per_event = np.count_nonzero(~np.isnan(el_pt_forCount), axis=1)
    for i, num_electrons in enumerate(num_electrons_per_event): #replace 3 with 2 in the presence of electrons
        lep_id[i, :num_electrons] = 2

    lep_id[lep_e == 0] = 0 #empty events are assigned an id of 0

    # Met features
    met_phi   = tree["Met_phi"].array(library='np')
    met_pt    = tree["Met_pt"].array(library='np')

    # Global features (eventNumber only used for the splitting)
    event_number = tree["eventNumber"].array(library='np')
    njet         = (ak.count(tree["Jet_pt"].array(),axis=1).to_numpy()).reshape(-1,1)
    nlep         = np.count_nonzero(~np.isnan(lep_pt), axis=1)
    nbTagged_100 = tree["nBjets_GN2v01_100_NOSYS"].array(library='np')
    nbTagged_90  = tree["nBjets_GN2v01_90_NOSYS"].array(library='np')
    nbTagged_85  = tree["nBjets_GN2v01_85_NOSYS"].array(library='np')
    nbTagged_77  = tree["nBjets_GN2v01_77_NOSYS"].array(library='np')
    nbTagged_70  = tree["nBjets_GN2v01_70_NOSYS"].array(library='np')
    nbTagged_65  = tree["nBjets_GN2v01_65_NOSYS"].array(library='np')

    # Labels (from parton truth)
    up_index = tree["Truth_up_index"].array(library='np')
    down_index = tree["Truth_down_index"].array(library='np')
    bhad_index = tree["Truth_had_b_index"].array(library='np')
    blep_index = tree["Truth_lep_b_index"].array(library='np')

    # ---------------------------------------------------
    #               Count number of events to save
    # ---------------------------------------------------

    num_events_to_save = num_entries if not split else sum(
        (event_number[i] % 2 == 0) == saveEven for i in range(num_entries)
    )

    # ---------------------------------------------------
    #               Define Input Datatype
    # ---------------------------------------------------


    node_dt = np.dtype([('e', np.float32), ('eta', np.float32), ('phi', np.float32), ('pt', np.float32), ('btag', np.float32), ('charge', np.float32), ('id', np.float32)])


    jet_data = np.full((num_events_to_save, pad_to_jet), np.nan, dtype=node_dt)
    lep_data = np.full((num_events_to_save, pad_to_lepton), np.nan, dtype=node_dt)
    met_data = np.full((num_events_to_save, 1), np.nan, dtype=node_dt)

    #Define global var
    global_dt = np.dtype([('njet', np.float32), ('nBJets_90', np.float32), ('nBJets_85', np.float32), ('nBJets_77', np.float32), ('nBJets_70', np.float32), ('nBJets_65', np.float32)])
    
    global_data = np.zeros((num_events_to_save, 1), dtype=global_dt)


    # We need `VertexID` to label truth edges and hyperedges
    jet_labels      = np.full((num_events_to_save, pad_to_jet), np.nan)  # use -9 for none filled values
    lepton_labels   = np.full((num_events_to_save, pad_to_lepton), np.nan)
    met_labels      = np.full((num_events_to_save, 1), np.nan)

    # ---------------------------------------------------
    #               Define HDF5 Structure
    # ---------------------------------------------------
    h5_file = h5py.File(output, 'w')
    inputs_group = h5_file.create_group('INPUTS')
    labels_group = h5_file.create_group('LABELS')

    print('Saving %d events' % num_events_to_save)
    save_index = 0
    for i in tqdm(range(num_entries), desc='Creating HDF5 dataset', unit='events'):
        
        #Check if event should be saved
        if split and ((event_number[i] % 2 == 0) != saveEven):
            continue

        #If there is a limit on the number of events to save, check it
        if save_index >= num_events_to_save:
            break
                
        num_jets = njet[i][0]
        for j in range(num_jets):
            jet_data[save_index][j] = (
                jet_e[i][j], jet_eta[i][j], jet_phi[i][j], jet_pt[i][j],
                jet_bTag[i][j], 0, 1
            )

        #Assuming only one lepton
        num_lep = 1
        lep_data[save_index][0] = (
            lep_e[i][0], lep_eta[i][0], lep_phi[i][0], lep_pt[i][0],
            0, lep_q[i][0], lep_id[i][0]   # Filling in 0 for the btag
        )

        met_data[save_index][0] = (
            met_pt[i], 0, met_phi[i], met_pt[i], 0, 0, 4
        )
            
        global_data[save_index] = (njet[i], nbTagged_90[i], nbTagged_85[i], nbTagged_77[i], nbTagged_70[i], nbTagged_65[i])

        existent_jet_labels = np.array([0] * num_jets + [np.nan] * (pad_to_jet - num_jets))
        existent_lep_labels = np.array([0] * num_lep  + [np.nan] * (pad_to_lepton - num_lep))
        existent_met_labels = np.array([0] * 1)
        
        if up_index[i] != -1:
            existent_jet_labels[up_index[i]] = 3
        if down_index[i] != -1:
            existent_jet_labels[down_index[i]] = 4
        if bhad_index[i] != -1:
            existent_jet_labels[bhad_index[i]] = 2
        if blep_index[i] != -1:
            existent_jet_labels[blep_index[i]] = 1
        existent_lep_labels[0] = 1
        existent_met_labels[0] = 1

        jet_labels[save_index]      = existent_jet_labels
        lepton_labels[save_index]   = existent_lep_labels
        met_labels[save_index]      = existent_met_labels

        save_index += 1

    #Inputs h5 datasets
    inputs_group.create_dataset("JET",     data=jet_data)
    inputs_group.create_dataset("LEPTON",  data=lep_data)
    inputs_group.create_dataset("MET",  data=met_data)
    inputs_group.create_dataset("GLOBAL", data=global_data)

    #Labels h5 datasets
    labels_group.create_dataset("JET", data=jet_labels) 
    labels_group.create_dataset("LEPTON", data=lepton_labels)
    labels_group.create_dataset("MET", data=met_labels)

    # Close the HDF5 file
    h5_file.close()
    root_file.close()
```