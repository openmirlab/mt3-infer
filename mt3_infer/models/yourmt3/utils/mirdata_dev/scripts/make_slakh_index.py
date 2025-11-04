"""make_slakh16k_index.py
USAGE:
    python tasks/utils/mirdata_dev/scripts/make_slakh_index.py '../data' '2100-yourmt3-16k'

"""
import argparse
import glob
import json
import os
import yaml
from mirdata.validate import md5


def get_file_info(path):
    if os.path.exists(path):
        return [path, md5(path)]
    else:
        print("warning: {} not found. check metadata for omitted files.".format(
            path))
        return [None, None]


def make_dataset_index(dataset_data_path, version):
    curr_dir = os.getcwd()
    os.chdir(dataset_data_path)

    dataset_index_path = os.path.join(dataset_data_path, "mirdata_indexes",
                                      f"slakh_index_{version}.json")

    if version == "baby":
        splits = [""]
        topdir = "babyslakh_16k"
        fmt = "wav"
    elif version == "2100-yourmt3-16k":
        splits = ["train", "validation", "test"]
        topdir = "slakh2100_yourmt3_16k"
        fmt = "wav"
    elif version == "2100-redux":
        splits = ["train", "validation", "test", "omitted"]
        topdir = "slakh2100_flac_redux"
        fmt = "flac"
    multitrack_index = {}
    track_index = {}

    for split in splits:
        mtrack_ids = sorted([
            os.path.basename(folder)
            for folder in glob.glob(os.path.join(topdir, split, "Track*"))
        ])
        for mtrack_id in mtrack_ids:
            print(f'indexing multitrack: {mtrack_id}')
            mtrack_path = os.path.join(topdir, split, mtrack_id)
            metadata_path = os.path.join(mtrack_path, "metadata.yaml")
            with open(metadata_path, "r") as fhandle:
                metadata = yaml.safe_load(fhandle)

            mtrack_midi_path = os.path.join(mtrack_path, "all_src.mid")
            mix_path = os.path.join(mtrack_path, "mix.{}".format(fmt))

            track_ids = []
            for track_id in metadata["stems"].keys():
                if metadata["stems"][track_id]["audio_rendered"] is not True:
                    continue  # <-- modified by @mimbres to avoid missing audio error
                if metadata["stems"][track_id]["midi_saved"] is not True:
                    continue  # <-- modified by @mimbres to avoid missing audio error
                audio_path = os.path.join(mtrack_path, "stems",
                                          "{}.{}".format(track_id, fmt))
                midi_path = os.path.join(mtrack_path, "MIDI",
                                         "{}.mid".format(track_id))
                midi_file_info = get_file_info(midi_path)
                # skip tracks where there is no midi information (and thus no audio)
                if midi_file_info[0] is None:
                    continue
                if get_file_info(audio_path)[0] is None:
                    continue  # <-- modified by @mimbres to avoid missing audio error
                track_id = "{}-{}".format(mtrack_id, track_id)
                track_ids.append(track_id)
                track_index[track_id] = {
                    "audio": get_file_info(audio_path),
                    "midi": [midi_file_info[0], midi_file_info[1]],
                    "metadata": get_file_info(metadata_path),
                }

            multitrack_index[mtrack_id] = {
                "tracks": track_ids,
                "midi": get_file_info(mtrack_midi_path),
                "mix": get_file_info(mix_path),
                "metadata": get_file_info(metadata_path),
            }

    # top-key level version
    dataset_index = {
        "version": version,
        "tracks": track_index,
        "multitracks": multitrack_index,
    }

    os.chdir(curr_dir)
    with open(dataset_index_path, "w") as fhandle:
        json.dump(dataset_index, fhandle, indent=2)


def main(args):
    make_dataset_index(args.dataset_data_path, args.version)
    print(
        f"A new index file is copied to {args.dataset_data_path}/mirdata_indexes/"
    )


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Make dataset index file.")
    PARSER.add_argument(
        "dataset_data_path", type=str, help="Path to dataset data folder.")
    PARSER.add_argument(
        "version",
        type=str,
        help="Dataset version. baby or 2100-redux or 2100-yourmt3-16k")
    main(PARSER.parse_args())