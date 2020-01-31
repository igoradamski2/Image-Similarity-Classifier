#!/bin/bash

while IFS="" read -r p || [ -n "$p" ]
do
	stringarray=($p)

	if [ -d "val/images/${stringarray[1]}" ]; then
		sudo mv ../tiny-imagenet-200/val/images/${stringarray[0]} ../tiny-imagenet-200/val/images/${stringarray[1]}
	fi
	
	if [ ! -d "val/images/${stringarray[1]}" ]; then
		sudo mkdir val/images/${stringarray[1]}
		sudo mv ../tiny-imagenet-200/val/images/${stringarray[0]} ../tiny-imagenet-200/val/images/${stringarray[1]}
	fi	
done < val/val_annotations.txt

for d in ../tiny-imagenet-200/val/images/* ; do
	cd $d
	sudo mkdir images/
	sudo mv *.JPEG images/
	cd ../../..	
done

cd ../tiny-imagenet-200/val/
sudo mv images/* .
sudo rm -r images/

