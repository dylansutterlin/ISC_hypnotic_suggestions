'''
Goals
- extract data from preprocessed folder, and relevant metadata
- plot combined carpet plot to see QC 
- Import the data
- Import the design matrices
- Specify the contrasts to run
- Run GLM and save results 

'''


# from nilearn.glm.first_level import FirstLevelModel

# for i, sub in enumerate(setup.subjects):

#     # Load 4D functional images and concatenate
#     ana_img = nib.load(setup.ana_run[i])  # Analgesia run
#     hyper_img = nib.load(setup.hyper_run[i])  # Hyperalgesia run
#     full_4d_img = concat_imgs([ana_img, hyper_img])  # Concatenate both runs

# # Initialize the GLM model (you probably already did this in your pipeline)
# fmri_glm = FirstLevelModel()  # Adjust t_r as per your TR value
# fmri_glm = fmri_glm.fit(setup.ana_run[i], design_matrices=dm_combined[i])  # Replace `fmri_img` with your actual fMRI data
