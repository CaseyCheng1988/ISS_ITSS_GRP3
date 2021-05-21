# Package Installation
___
* Create a `conda` environment:
	* `conda create -n movrating python=3.7.10`
* Activate the conda environment and install the following in order:
	* `conda install -c conda-forge librosa`
	* `pip install tensorflow==1.15.2`
	* `pip install -U get-video-properties`
	* `pip install moviepy`
	* `conda install -c pytorch pytorch=1.4.0`
	* `conda install -c conda-forge opencv`
	* `conda install pandas`
* CD into the *pyAudioAnalysis* folder and install using the following command:
	* `pip install -e .`
* If necessary, downgrade h5py version to below 3.0.0
	* `pip install h5py==2.10.0`

# Perform rating classification
* Launch the terminal inside the system folder.
* Run the system using the following command:
	* `python main.py -i <movie_path>`


