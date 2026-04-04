# Reproducibility instructions/tips

## General terminology/variables 

- 1 sample (bootstrap) = Test if ISC for one condition != 0
- Permutation : used for contrasts testing between conditions 


#### Model terminology
Behav univariate : pairwise similarity on univariate scores, e.g. SHSS
Cosine (multivariate) : Pairwise cosine similarity betwen multivariate vectors (e.g. SHSS, automat, focus, etc.)
ISC-sugg : Pairiwse ISC matrix during suggestions encoding
SHSS-behav : behavioral score only
ISC-sugg/pain : brain ISC for sugg or pain

#### Final model (hyper)parameters
- **Atlas** : Shaeffer200 + Tian 16 subcortical regions combined manually in main_isc.py   
    --> ID as `*_schafer_tian-200-2mm_*` in results names

### Manuscript plots from [scripts]
- Surface plots with parcel : `/visualization_isc.py`


## Visualization and final models
### ISC
`/visualization_isc.py` : Custom script to plot ISC per conditions, for median split (SHSS based), and permutation/contrast tests

Each ISC results contains a **setup_parametres.json** with necessary hyperparams, like conditions names, subjects, n_boot, used to further load results files 

INPUT : specify the proper model

    Final models found in /results/ISC/ include : 

    'model1_single-trial-wb_sugg_23-sub_schafer_tian-200-2mm_mask-whole-brain_pairWise-True_preproc_reg-mvmnt-True-8' 
        --> Model ran for each trial event (e.g. Neutral_sugg1, Neutral_sugg2, etc..), so produces ISC maps for each trials
        - Used to compared 1st neutral conditions of hyper vs ana run
        - Contrast between Ana-Neutral is acheived in this model. 
    
    'model2_sugg_23-sub_schafer_tian-200-2mm_mask-whole-brain_pairWise-True_preproc_reg-mvmnt-True-8',
        --> Main suggestion model with 216 ROIs    

    'model3_shock_23-sub_schafer_tian-200-2mm_mask-whole-brain_pairWise-True_preproc_reg-mvmnt-True-8',
        --> Main shock model with 216 ROIs

INPUT : Uses Setup 
OUPUT/CODE for : 
    - Saves output **results/figures/tables** in ad-hoc dir directly in results/ISC_model...
    - Saves surface atlas parcellation, !USed for publication
    - Calls function to visualize slices or ISC brain maps, input proper coords 
    - Plot heatmap of pariwise ISC maps
    - plot behavioral similarity maps 


# IS-RSA

## Suggestion (ISC)
INPUT : Pre-computed ISC models results path
    - Specified in model_names to fetch proper ISC model

- Does many specific tests : 
    - ISC (sugg / pain) ~ SHSS scores (univ.) / pain scores(univ.) / cosine-based multivariate
    - ISC-sugg ~ ISC-pain

OUTPUT : 
- !! there is a copy of the script inside output folder! 

FILE: `isc-RSA_ext_conds_sugg-pain_tian216_2tails` Final output for IS-RSA 
    - Based on pairwise ISC (schaeffer+Tian; 216 parcels) suggestion & pain models 
        - model_sugg = model_names['model2_sugg']
        - model_pain = model_names['model3_shock']

- pkl files names according to the model tested.
    - Contains [Euclidean, annak] model in dict with each conditions tested inside
        - Main conditions are ana_run, hyper_run, but also contain hyper, ana for comparison

#### Visualization
`visu_rsa_isc_ext_model.py` : main script to visualize ISC matrices, brain map of RSA 

INPUT : 

Main model (final) : 216 parcels, 2 sided test : RSA/isc-RSA_ext_conds_sugg-pain_tian216_2tails

OUTPUT : 
- Tables and thresholded maps save in ..model/post-hoc_VISU_tables

- ..model/post-hoc_figures/ : figurs used for publication

# Cross model IS-RSA : ISC-suggestion ~ contrast-based MVPA similarity pain

## IS-RSA pain (multivarie)
FILE: rsa_mvpa_from_glm_pain_variables.py
    - Extracts 1st level GLM contrast maps for `all_pain_conditions` (hyper>N_hyper & Ana>N_ANA maps)
    - Runs inter-subject multivoxel spatial similarity for each ROI
    - One sample test (median upper trig > 0)/ROI, for two pain conditions

    Hyperparams/: 
    - `NEURAL_SIM` : dot product and cosine sim. Dot product final, cosine for visualization. Models reproduce between the metric
    - `BEHAV_ID` : tests IS-RSA with SHSS_score, change in pain score for Analgesia and Hyperalgesi 

    ** Note :Change in pain Ana only tests with brain data from the Ana>N_Ana maps, and similarly for the hyper variable with the hyper maps. SHSS is tested with both brain conditions

    USE: Final SHSS model in paper. chge_pain_ana ~ PHG to confirm the implication in behavioral pain modulation.


//script overlaps with `rsa_mvpa_from_glm_dot.py` : initial script for SHSS ~ pain_MVPA rsa

### Visualization for `rsa_mvpa_from_glm_pain_variables.py`

`visualization_is-rsa_final.py` : produces publication plots for IS-RSA (dot sim., t maps) for MVPS pain modulation patterns ~ SHSS and ~ (behavioral sim. pain model)

(old earlier script V)
### Multivariate spatial IS-RSA from GLM

script use to procude and visualize: `rsa_mvpa_from_glm.py`
- saves in `MODEL_NAME` ('model3_final-isc_23subjects_nuis_nodrift_31-03-25') 
    adhoc_analyses_results
        - plots
            - PUBLICATION: glass brain GLM plots for suggestions 
        - is-rsa_patterns_similarity:
            - pairwise cosine pattern similarity per conditions

## IS-RSA (path 3) seed ISC with target IS-MVPA
FILE : rsa_isc_mvpa.py
    - Run 
    - Visu (manual hyperparam change)



# Visualization Published Manuscript:

## ISC

