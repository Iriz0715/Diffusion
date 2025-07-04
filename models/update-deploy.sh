# Note:
# For deploy build, all relevant scripts are "hard-linked" to the deployed algorithms so that they can be synced.
# That also means do not edit those files in each deployed algorithms!
# https://www.tecmint.com/create-hard-and-symbolic-links-in-linux/
# Use this script to update the hard links if necessary

#!/bin/bash
ln -f ./utils_detection_3d.py ../../quantlung-deployment/quantlung_analysis/utils_detection_3d.py