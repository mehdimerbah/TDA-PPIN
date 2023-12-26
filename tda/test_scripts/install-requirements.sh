#!/usr/bin/bash


cat requirements.txt | while read PACKAGE; do pip3 install "$PACKAGE"; done
