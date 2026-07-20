$VenvPath = Join-Path $PSScriptRoot "..\.venv"
$ActivationPath = "$VenvPath\Scripts\Activate.ps1"
$NSPackagesFile = Join-Path $PSScriptRoot "ns-packages.txt"
$RequirementsPath = Join-Path $PSScriptRoot "requirements.txt"


# check if .venv exists
if (Test-path $VenvPath) {
    Write-Output "Virtual environment found at $VenvPath"
}
else {
    Write-Output "Virtual environment not found. Creating at $VenvPath"
    # create a venv
    python -m venv $VenvPath
    Write-Output "Virtual environment created at $VenvPath"
}

# Activate the venv
Write-Output "Activating venv"
& $ActivationPath

# Check if the 'ns-packages.txt' file exists
if (Test-Path $NSPackagesFile) {
    Write-Output "$NSPackagesFile found, Installing packages..."
    pip install -r $NSPackagesFile
} else {
    Write-Output "$NSPackagesFile not found! Please ensure it is located in the project root if you require these packages."
}

# ask the user if they wish to install requirements.txt
if (Test-Path -LiteralPath $RequirementsPath){ 
    $InstallRequirements = Read-Host "Found requirements.txt. Do you wish to install packages from 'requirements.txt'? (y/n)"
    if ($InstallRequirements -eq 'y' -or $InstallRequirements -eq 'Y') {
        Write-Output "Installing $RequirementsPath"
        pip install -r $RequirementsPath
    }
    else {
        
        Write-Output "Skipping installation of packages from requirements.txt"
    }
}
else {
    Write-Output "$RequirementsPath not found! Reminder to 'pip freeze > $RequirementsPath' when you have installed some packages to keep your environment up to date."
}