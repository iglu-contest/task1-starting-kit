Data
====

### A Warm-up dataset

This dataset is from the ACL 2019 paper [Collaborative Dialogue in Minecraft](http://juliahmr.cs.illinois.edu/Minecraft/ACL2019.html). The Minecraft Dialogue Corpus (without screenshots) is stored in `data/logs`. The target structures are in `gold-configurations`. `logs/splits.json` defines the train-test-val split by defining subsets of target structures. You can download the corpus without screenshots [here](https://drive.google.com/file/d/1p5_SbFywFBHiKDx2M8xfhlimg6Yu_0IP/view?usp=sharing), the corpus with screenshots [here](https://drive.google.com/file/d/1fUIiQydVYEe-IQZqNNVTtenTGljNOrah/view?usp=sharing).

### IGLU dataset (coming soon)

This dataset is an extension of the original [Minecraft Dialogue Corpus](http://juliahmr.cs.illinois.edu/Minecraft/ACL2019.html). We have extended the original dataset by adding new structures and corresponding dialogues. We merged the original Minecraft Dialogue Corpus with our new dataset and resplit them to `train`, `val`, and `test` sets. It is guaranteed that the structures and dialogues in the `test` set are not included in the original Minecraft Dialogue Corpus. The `test` set is unseen to all the participants during the competition.

#### \* Teams do not need to adjust their own dataloader while the new IGLU dataset is released. We will make sure UIUC data and IGLU data have the same data structure (more details below) and data format.

####

Data structure
--------------

The UIUC data is organized as follows:

    uiuc_warmup
        ├── Data_format.docx  # data structure information provided by the UIUC team
        ├── gold-configurations  # the target structures 
        ├── splits.json     # this file defines the train-test-val split by defining subsets of target structures. We do not hide the test set information because this dataset is publicly available. Thanks to all the authors for releasing it. 
        ├── data-x-xx        # all sessions collected on the same day will be stored in the same folder
            ├── logs        # folder 'logs' stores textual information of sessions.
                ├── Bxx-Axx-Cx-xxxxxxxxxxxx1 #  session ID
                    ├── postprocessed-observations.json  
                    |── aligned-observations.json
                ├── Bxx-Axx-Cx-xxxxxxxxxxxx2 #  session ID
                    ├── postprocessed-observations.json  
                    |── aligned-observations.json
                ├── ...
            ├── screenshots   # all screenshots stored during the game, organized by session ID (to save space, we have reduced the image size).
                ├── Bxx-Axx-Cx-xxxxxxxxxxxx1 #  session ID
                    ├── xxx.jpg  
                    |── xxx.jpg   
                    ├── ...
                ├── Bxx-Axx-Cx-xxxxxxxxxxxx2 #  session ID
                    ├── xxx.jpg   
                    |── xxx.jpg   
                    ├── ...
                ├── ...
    
        ...
    
        ├── data-x-xx        # all sessions collected on the same day will be stored in the same folder
            ├── logs        # folder 'logs' stores textual information of sessions.
                ├── Bxx-Axx-Cx-xxxxxxxxxxxx1 #  session ID
                    ├── postprocessed-observations.json  
                    |── aligned-observations.json
                    ├── ...
                ├── Bxx-Axx-Cx-xxxxxxxxxxxx2 #  session ID
                    ├── postprocessed-observations.json  
                    |── aligned-observations.json
                    ├── ...
                ├── ...
            ├── screenshots    # all screenshots stored during the game, organized by session ID.
                ├── Bxx-Axx-Cx-xxxxxxxxxxxx1 #  session ID
                    ├── xxx.jpg  
                    |── xxx.jpg   
                    ├── ...
                ├── Bxx-Axx-Cx-xxxxxxxxxxxx2 #  session ID
                    ├── xxx.jpg   
                    |── xxx.jpg   
                    ├── ...
                ├── ...
    

#### aligned\_observations.json: this is the main file to access the log data collected. Its format is as follows (copied from Data\_format.docx):

    Basically, this is a chronological list of all observations. An observation gets recorded when:
    -When the builder places a block
    -When the builder picks up a block
    -When a chat message is sent by either architect or builder
    
    The top-level dictionary:
    {
        WorldStates: List of all world states recorded in chronological order (more below)
    
        NumFixedViewers: Number of fixed view perspectives used to collect additional screenshots (always 4 for us) apart from the architect and builder perspectives
    
        TimeElapsed: Total time taken for the game from start to finish in seconds
    }
    
    WorldStates: Each item in the list is the world state at a certain point in time. It is as follows:
    
    {
        TimeStamp: The exact time when this observation was recorded
    
        BuilderPosition: Builder’s <x, y, z, yaw, pitch> coordinates
    
        BuilderInventory: List containing 6 items corresponding to all 6 block colors available in the game. For each color, it stores the number of blocks of that color that the builder currently possesses.
    
    BlocksInGrid: All blocks currently placed in the build region. Each block has a color and 2 sets of <x, y, z> coordinates -- absolute and perspective. The latter is relative to the builder’s perspective, i.e, the builder’s <x, y, z, yaw, pitch> coordinates (recorded in BuilderPosition above).
    The build region is an 11 x 9 x 11 grid, where -5 <= x <= 5,  -5 <= z <= 5 and 1 <= y <= 9 (where y=1 is ground level).  (Important! y is the 3D vertical axis in the Minecraft world!)
    
        ChatHistory: The entire dialog history up until this point in time
    
        Screenshots: All screenshots taken for this observation. There are 6 fields for the 6 screenshot perspectives -- builder, architect, and the 4 fixed viewers. Each has a value which is the name of the screenshot image in the corresponding screenshots directory. In the case where there was no screenshot taken or was taken but couldn’t be aligned to this observation, the value for the corresponding field is null.
    
        DialogueStates (**Builder demo only**): If the dialogue manager was involved in directing the flow of information through the system at that point in time, the DialogueStates field will contain a list of dialogue states that the system transitioned through as well as their arguments (specific to the dialogue state; i.e., a resulting plan from a PLAN state, or a semantic parse from a PARSE state). The information consists of: the name of the Dialogue State (e.g. PARSE_DESCRIPTION, PLAN, REQUEST_VERIFICATION, etc.); the input text (from the human Architect); the output text (from the system as Builder); the resulting semantic parse produced by the parser; the response from the planner during planning; and the execution status of the provided plan.
    
    }
    

#### Important: For the sessions in IGLU data, we will only provide screenshots from the Builder's view together with screenshots of finished target structures. This is also the setup for the final competition evaluation. In UIUC data, 4 additional fixed views are provided.

Participation in IGLU-task-1
============================

You can either extend the released models from paper [Collaborative Dialogue in Minecraft](http://juliahmr.cs.illinois.edu/Minecraft/ACL2019.html) or develop your own experiment pipelines. We will give an introduction about how to adapt a developed model to a runnable IGLU submission. 

### Dataloader:

1.  One dialogue session can be grouped into different context-reply pairs. To let teams make full use of the released dialogues, there is no restriction on how the dialogue context is organized. For example, you could treat the last $m$ dialogue turns as your dialogue context and the value of $m$ is decided by yourself. Each architect reply in dialogue has a unique $time\_stamp$ and we use it to distinguish different architect replies.
2.  There is a possibility that several sequential architect replies may have the same $time\_stamp$ because the speaker sends messages too fast. To avoid ambiguity, you are asked to predict the first architect reply in this dialogue segment and skip others while generating predictions.
3.  It is guaranteed the `test` set has the same data structure as the `train` and `val` sets. After you submit your submission, we will feed the `test` set together with $output\_path$ to your model via a reserved function, `model.generate(dataset, output_path)`. You need to make sure that your dataloaders can load the `val` set successfully.
4.  We expect to obtain an output file by running the function `model.generate(dataset, output_path)`. The output file is supposed to have two columns, separated by the special string `@@@`. The left column is $time\_stamp$ and the right one is the corresponding prediction. You can find more submission requirements below.

### How to make a submission

1.  There should be a file named `generate_iglu.py` in the root directory of your submission. You are asked to implement one function and one class in this file: function `create_dataloader` and class `Model()`. More details can be found in the next section.
2.  Your trained model should be stored in your submission folder and can be loaded by your own model. You are asked to implement the loading step.
3.  If your code uses external packages beyond the existing docker environment, please provide the docker image id in the file named `Dockerimage` in the root directory of your submission. If `Dockerimage` does not exist in the root directory, our evaluation pipeline will load the default docker image: `zimli/iglu-task1:v4`. During the evaluation, the docker environment has no access to the network and you should pre-download all materials to the docker environment you provide.
4.  Make sure you follow all the submission requirements.
5.  Zip the system and submit.

### Submission code structure

Please follow the following code structure.

    IGLU-task-1
    ├── * metadata               # Submission meta information, including your username/team-name.
    ├── * Dockerimage            # Place the docker image id in the first line. This is a single-line file.
    ├── * generate_iglu.py       # The entry point for the testing phase.
    ├── /saved_models            # This directory can store your trained models.
    ├── /models                  # Your developed model code
    └── /tools                   # The utility scripts which provide a smoother experience to you.
    

Files with \* are essential in the submitted code. Others are optional. In file `generate_iglu.py`, you are asked to implement one function, `create_dataloader` and one class`Model()`.

    def create_dataloader(data_path, split):
        # data_path: the root path to the data, e.g. '/datadrive/uiuc_warmup/'
        data = ...YOUR CODE HERE...
        return data
    

    class Model():
        def __init__(self):
            # The initialization function does not take input. The output of this function should be a class object of your model. 
            ...YOUR CODE HERE...
            return model
    
        def generate(self, test_set, output_path):
            # test_set: the dataloader object created by your function: create_dataloader
            # output_path: the path to a '.txt' file that stores your predictions. This file is supposed to have two columns, separated by the special string @@@. The left column is time_stamp and the right one is the corresponding prediction. 
            ...YOUR CODE HERE...
    

An example submission
============

#### In the following sections, we will show how to convert baseline models from paper [Collaborative Dialogue in Minecraft](http://juliahmr.cs.illinois.edu/Minecraft/ACL2019.html) to a submission of IGLU task-1. 

Environment setup
-----------------

*   [PyTorch](https://pytorch.org/get-started/locally/) (`v1.0.1` is validated)
*   [NLTK](https://www.nltk.org/install.html)
*   Python 3.7
*   Numpy
    
### Docker
    
    docker pull zimli/iglu-task1:v4
        
    

Code structure
--------------

In directory `python`, there are a number of models implemented in the file `model.py` file -- `utterances_and_next_actions, utterances_and_block_counters, utterances_and_block_region_counters, image_cnn_utterances_and_block_region_counters, seq2seq_all_inputs, seq2seq_attn, seq2seq_world_state`.

If you want to develop models by entending these baselines, you can follow similar coding patterns and add them to the appropriate directory's `model.py` or create a new such directory if needed. We use `utterances_and_block_region_counters` as an example in the following part.

### Dataloader

1.  The provided dataloader first collects all JSON files given the input data path and split (function `get_logfiles_with_gold_config(...)`). This function can be reused in most cases.
2.  In the second step, the retrieved JSON files are processed to generate a list of data samples (function `process_samples(...)`). Each data sample corresponds to a specific context-reply pair with additional session information, such as `time_stamp`. This step could be customized according to your own input needs.
3.  After a list of data samples have been obtained, each context-reply pair will be converted to tensor representations in function `__getitem__(...)`. In the baseline models from the paper [Collaborative Dialogue in Minecraft](http://juliahmr.cs.illinois.edu/Minecraft/ACL2019.html), global structure difference and local region difference are calculated. You may want to adjust this step so it can provide your own models with informative information.
4.  In the last step, a mini-batch of data items are returned by function `collate_fn(...)`

### Training

this is the script to train a prediction model, `utterances_and_block_region_counters`:

    python -u trainer.py utterances_and_block_region_counters /home/iglu/minecraft-architect/your/config/path/best_model.config
    

In this script, you need to change the config path based on your own code structure. After you download the dataset, you should replace the relevant fields (e.g., `model_path `, `saved_dataset_dir`, `data_dir `) with your own data path. `utterances_and_block_region_counters` is the model name. After the training is done, models will be saved to path `model_path` + `a unique folder created during training`. The training log will give you all the relevant information.

**Note:** During training stage, you can feed args to the main function of trainer.py with two different ways: 1. you predefine a config file (`parser.add_argument('hyperparameter_file', type=str, help='file of hyperparameter options to train models for')`) including all the hyperparams you want to test; 2. if you do not want to use a predefined config file, you can set the values through parser. If you have used both the config file and parser to set your hyperparams, the values in the config file will overwrite the values in parser. For example, if you don’t want to hardcode the value of `--model_path` in the config file, you should remove this field from the config file and feed your desired path using `python trainer.py --model_path /your/own/path`. For the final submission, all hyperparams should be loaded from a config file or be hardcoded in your code. 
### Testing

If you want to get predictions on a specific set, you can run

    python -u generate_seq2seq.py /your/path/to/saved.model --beam_size 10 --gamma 0.8 --saved_dataset_dir /your/path/to/processed/dataset
    

To test if your model is ready to submit, you can run:

    python -u generate_iglu.py
    
In this example submission, we already placed a trained model in the directory `./saved_models` together with the corresponding config file `saved_model/1626589670356/config.txt`. Don't forget to change the datapath in the main function of `generate_iglu.py`. For your own submissions, you should load all relevant hyper-params from a config file or hardcode them in the code. 

### About the submission

To convert a developed model to an IGLU submission, you just need to implement function `create_dataloader` and class `Model()` in file `generate_iglu.py`. After this has been finished, you can run `python -u generate_iglu.py` to check if you have a runnable submission as we mentioned above. It's also crucial to test your model in the docker environment you provided and the environment will not have access to the internet.

### More details about the baseline models

If you prefer to extend the models presented in paper [Collaborative Dialogue in Minecraft](http://juliahmr.cs.illinois.edu/Minecraft/ACL2019.html), you may want to carefully go through their [github page](https://github.com/prashant-jayan21/minecraft-dialogue-models).