#!/bin/bash

set -e
# fbo-nightly.sh [YYYYMMDD]
# download the nightly file
# get/require a date. yesterday by default.
if [[ $# -eq 0 ]]
then
    set  +e
	date --version >/dev/null 2>&1
	# check return code
	if [[ $? -eq 0 ]]
	then
		# GNU date format
		download_date=`TZ=US/Eastern date --date yesterday +"%Y%m%d"`
	else
		# try this instead
		download_date=$(TZ=US/Eastern date -v -1d +"%Y%m%d")
	fi
    set -e
elif [[ $1 -ne "" ]]
then
    set +e 
	date --version >/dev/null 2>&1
    if [[ $? -eq 'gnu' ]]
    then
        date -d $1
    else
        gdate -d $1
    fi

	if [[ $? -eq 1 ]]
	then
		echo "Usage: fbo-nightly.sh [YYYYMMDD]"
	else
		download_date=$1
	fi
    set -e
fi

echo "Starting fbo-nightly $download_date"

FBOPEN_URI=${FBOPEN_URI:-"localhost:9200"}
echo "FBOPEN_URI = $FBOPEN_URI"
FBOPEN_INDEX=${FBOPEN_INDEX:-"fbopen"}
echo "FBOPEN_INDEX = $FBOPEN_INDEX"

# always change to this script's own directory
cd "$(dirname "$0")"
echo "PWD: "; echo $PWD

# mkdir -p will ensure the workfiles dir is in place, but won't fail if it already exists
workfiles_dir="$PWD/workfiles"
mkdir -p $workfiles_dir

# download the nightly file, if not downloaded already
nightly_download_file="$workfiles_dir/FBOFeed$download_date.txt"
echo "nightly download = $nightly_download_file"

if [[ ! (-s $nightly_download_file) ]]
then
	wget -O $nightly_download_file ftp://ftp.fbo.gov/FBOFeed$download_date
else
	echo "(already downloaded $nightly_download_file)"
fi

echo "converting nightly file into JSON"
# process the nightly file into JSON
cat $nightly_download_file | node $PWD/xml2json.js > $nightly_download_file.json

# prep the JSON further
prepped_json_notices_file=$workfiles_dir/prepped_notices.$download_date.json
cat $nightly_download_file.json | node $PWD/process_notices.js > $prepped_json_notices_file



