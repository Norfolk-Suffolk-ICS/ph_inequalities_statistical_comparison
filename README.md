# Project introduction 

This is a clean, minimal template repository for starting new Python projects in SCC's remote desktop, with sensible defaults for structure, environment setup and packaging. This template has a suggested project structure, it has 3 main directories:

- **Data**: This is where all your data for analysis will/should be stored.
- **Notebooks**: This is where all your Python/R notebooks for analysis should be stored.
- **Setup**: This folder holds all the files related to python setup and snowflake integration.

> ⚠️ Note - Please do no rename or move (change location) of the 'Setup' folder and any files within the folder. 
<br>

## Step to set up snowflake extension on VScode 

**(To be done only once, the very first time you setup VScode and Snowflake)**

1. Go to extensions from left most bar on your VScode screen and click on Extensions
2. Search for 'snowflake' and install.
3. Once installed, paste in your accoutn identified/URL (You can find this in your account details on SNOWFLAKE) and click continue.
4. Select 'Single-sign-on' as Auth Method.
5. Type in your username/email address used to login into snowflake and you are good to do.
6. You should see option like Databases and applications on your left bottom panel, similar to what you see on snowflake from browser.


## Steps for python setup

**(To be done each time you create a new repository using this template)**

1. In Vscode, on the left-hand tab, click "Explorer". Go to ```Setup``` directory and open the file ```connections.toml.example```
2. Please rename the file to ```connections.toml``` , this will ensure your personal details (email) will not be pushed to github.
3. Replace both the ```user``` fields with your datahub/snowflake email address and save.
4. Now again go to ```Setup``` directory and open the file ```setup-env.ps1```
5. At the top right of the powershell file click the play button, this will execute this powershell script.
6. Wait and keep an eye on your terminal window to check status, type in 'y' if it promts to download packages from ```requirements.txt``` file.
7. Once it stop running your python setup is finished and you can start on your project.


## Virtual environment

Is is always recommended to use a virtual environment in any analytical product.
This allows another person to replicate, review, contribute to, and run your project using the same underlying code packages.


## Setup directory explained

This directory contains all the files you need for python setup and integration with Snowflake.

1. It contains a powershell script file i.e. 'setup-env.ps1', which is core set-up file, it:
   - Checks for virtual environment at ```.\.venv```
      - If one is found. it is activated.
      - If not found, one is created, then activated.
   - Checks for ```ns-packages.txt``` and attempts installation. See below for details.
   - Checks for ```requirements.txt``` and prompts the user to install the packages within, if it is present.


2. It also contains 'connection.toml' file : This information in this file is used to integrate snowflake and use dataset stored in snowflake directly in VS code.

> ⚠️ Note - Please update the 'user' field in the file with your username/email.


## NS Packages

By default, this installs NS_utils package when you run the setup-env.ps1 file :

- NS_utils is an Internal Python Library (Python package distribution) designed for reuse by analysts within the Norfolk and Suffolk (NS) Intelligence Function. For more info read the README.md here --> [NS utils](https://github.com/Norfolk-Suffolk-ICS/NS_utils/blob/main/README.md)


## Explaining the files

1. ```.gitignore```        --> This is a plain text file that tells Git which files or directories to ignore, meaning they wont be tracked, committed or pushed
2. ```README.md```         --> Readme file for the repository (Not related to python setup). The contents of this file will be displayd on GITEA repository page
3. ``` connection.toml```  --> toml file to establish connection with snowflake.
4. ```requirements.txt```  --> This file is used to list and download the basic Python packages your project would depends on.
5. ```setup-env.ps1```     --> This file is the main setup file. It has 3 important steps
   - Check for Virtual environment: if found then skip else create new VE (.venv) and activate it.
   - Check for any repositories you want to load from ns-packages.txt file (by default it has NS-IF stylings and python functions, you can add on your)
   - Download all the Python packages from requirements.txt file
6. ```ns-packages.txt``` --> This file lists all the repositories/packages we want to load from Gitea.
7. ```example_analysis.ipynb``` --> This is an example fiel on how to establish connection with snowflake and load data from sql, load data from a file and NS plotting themes.

<hr>
