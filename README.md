# NEWAXOVO
Ensemble approaches are methods that aggregate the output of different (base) classifiers to achieve the final classification outcome. 
The diversity of the base classifiers is key to improving the effectiveness and robustness of the recognition performance. 
A very well-known approach to differentiate the pool of base classifiers is applying a one-vs-one decomposition schema, i.e. decomposing the classification problem into many binary classification problems, one for each pair of classes. One-vs-one decomposition schemas can be affected by the problem of non-competent classifiers. 
A base classifier is non-competent for the classification of a sample if its class differs from the pair of classes used for training the base classifier. In this case, the base classifier’s outcome is unreliable and may deteriorate the recognition performance. 
Moreover, with ensemble approaches the explainability of the final prediction is non-trivial and requires an ad-hoc design. 
We present NEWAXOVO, an ensemble method based on one-vs-one decomposition schemas that is capable of handling non-competent classifiers and provides contrastive explanations for its predictions. 
This architecture is designed for recognizing motor imagery from electroencephalogram (EEG) data and tested on a real-world dataset. 
We also compare NEWAXOVO with similar architectures proposed in the literature, such as DRCW-OVO, and we show that NEWAXOVO outperforms them under the considered metrics.

> The software here provided is the Python implementation of NEWAXOVO. The software is provided “as is”, without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose, and noninfringement. In no event shall the authors or copyright holders be liable for any claim, damages, or other liability, whether in an action of contract, tort, or otherwise, arising from, out of, or in connection with the software or the use or other dealings in the software.


# REPO CONTENTS
This release is provided as a single project (implemented in Pycharm) along with a folder organization we recommend not to modify.

1. The "data" folder contains the first dataset used in the study in ".csv" format.

2. The folder "src" contains the source of the implementation and it is organized in the following subfolders:
* "DataGenerators," "Metrics," and "Utils" contain utility functionality. 
* NAWAX: contains OneVsOne version (NEWAXOVO) and OneVsAll version (NEWAXOVA) of the new multiclass decomposition scheme with clustering-based weighing
* DRCW: contains the OneVsOne (NEWAXOVO) version and the OneVsAll (NEWAXOVA) version of the multiclass decomposition scheme based on 
* TAWAX: contains the OneVsOne (Unweighted OVO) and the OneVsAll (Unweighted OVA) version of the classic multiclass decomposition scheme usually used in ensembles

3. The "results", "images" and "explanations" folders will contain the results of the computation according to their type

4. The "test" folder contains 2 main files that illustrate the basic usage of the proposed approach:
* "processingHistoryPredictMain.py" allows the analysis of the data contained in the "data" folder to derive both the recognition of the imagined movements and the computation of the metrics used in the study. The results will be printed on the screen and saved in appropriate subfolders in "historyPredictions" in ".csv" format
* "explanationNeighborsMain.py" allows you to generate examples of model explanations that will be saved in "images/plots" if graphical and in "explanations" if in text format


# HOW TO USE
It is possible to use the provided main files to get an example of a possible execution of the release.

Before using the main files you need to:
1. install the requirements listed in "requirements.txt"
2. modify the "base_paths" in the code according to the local path where the project is downloaded. The "base_path" variables are present in the main files, in the "NAWAX.py", "DRCW.py" and "TAWAX.py" classes, and in "Utils/Plots.py"
3. Set the main variables appropriately:
* subject_ids => are the IDs of the users in the data (do not need to be modified if you are using the provided data)
* base_path => local path to the project
* n_classes => number of classes (e.g., imagined movements) to be recognized 
* n_neighbors => the number of neighbors to be used to weight the outputs of the base classifiers with NEWAXOVO and DRCWOVO
* configuration => "1vs1" or "1vsALL" depending on whether an OVO or OVA scheme is desired

> More information on all the components of the release can be found in the file "user_manual.pdf"