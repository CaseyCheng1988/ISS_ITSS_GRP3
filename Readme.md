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
* CD into the *pyAudioAnalysis* folder and install using the following command:
	* `pip install -e .`

# Perform rating classification
* Launch the terminal inside the system folder.
* Run the system using the following command:
	* `python main.py -i <movie_path>`


