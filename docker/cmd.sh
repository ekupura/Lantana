#!/bin/sh

python3 -m unidic download
pip3 freeze > /mnt/requirements.txt
pip3 install -U nbconvert

/usr/sbin/sshd -D

/bin/bash
