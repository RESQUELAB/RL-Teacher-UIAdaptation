import os
import multiprocessing
import pickle
import bisect
from time import sleep

import numpy as np

from rl_teacher.video import write_segment_to_video, upload_to_gcs

from django.db.models import Max


def _write_and_upload_video(clip_data, frames, clip_id, source, fps, gcs_path, video_local_path, clip_local_path, actions=""):
    with open(clip_local_path, 'wb') as f:
        pickle.dump(clip_data, f)  # Write clip to disk
    write_segment_to_video(frames, fname=video_local_path, fps=fps)
    # upload_to_gcs(video_local_path, gcs_path)
    return clip_id, source, actions

def _tree_minimum(node):
    while node.left:
        node = node.left
    return node

def _tree_successor(node):
    # If we can descend, do the minimal descent
    if node.right:
        return _tree_minimum(node.right)
    # Else backtrack to either the root or the nearest point where descent is possible
    while node.parent and node == node.parent.right:
        node = node.parent
    # If we've backtracked to the root return None, else node.parent will be successor
    return node.parent


class SynthClipManager(object):
    """Like the basic ClipManager, but uses the original environment reward to sort the clips, and doesn't save/load from disk/database"""

    def __init__(self, env, experiment_name):
        self.env = env
        self.experiment_name = experiment_name
        self._sorted_clips = []  # List of lists of clips (each sublist's clips have equal reward sums)
        self._ordinal_rewards = []  # List of the reward sums for each sublist

    def add(self, new_clip, *, source="", sync=False):
        # Clips are sorted as they're added
        new_reward = sum(new_clip["original_rewards"])
        if new_reward in self._ordinal_rewards:
            index = self._ordinal_rewards.index(new_reward)
            self._sorted_clips[index].append(new_clip)
        else:
            index = bisect.bisect(self._ordinal_rewards, new_reward)
            self._ordinal_rewards.insert(index, new_reward)
            self._sorted_clips.insert(index, [new_clip])

    @property
    def total_number_of_clips(self):
        return self.number_of_sorted_clips

    @property
    def number_of_sorted_clips(self):
        return sum([len(self._sorted_clips[i]) for i in range(len(self._sorted_clips))])

    @property
    def maximum_ordinal(self):
        return len(self._sorted_clips) - 1

    def sort_clips(self, wait_until_database_fully_sorted=False):
        """Does nothing. Clips are sorted as they're added."""

    def get_sorted_clips(self, *, batch_size=None):
        clips_with_ordinals = []
        for ordinal in range(len(self._sorted_clips)):
            clips_with_ordinals += [{'clip': clip, 'ord': ordinal} for clip in self._sorted_clips[ordinal]]
        sample = np.random.choice(clips_with_ordinals, batch_size, replace=False) if batch_size else clips_with_ordinals
        return [-1 for _ in sample], [item['clip'] for item in sample], [item['ord'] for item in sample]


class ClipManager(object):
    """Saves/loads clips from disk, gets new ones from teacher, and syncs everything up with the database"""

    def __init__(self, env, env_id, experiment_name, workers=4, user_id=None, domain=None, training=True):
        # self.gcs_bucket = os.environ.get('RL_TEACHER_GCS_BUCKET')
        # assert self.gcs_bucket, "you must specify a RL_TEACHER_GCS_BUCKET environment variable"
        # assert self.gcs_bucket.startswith("gs://"), "env variable RL_TEACHER_GCS_BUCKET must start with gs://"

        self.env = env
        self.env_id = env_id
        self.experiment_name = experiment_name
        print("---------------")
        print("---------------")
        if domain:
            print("Creating the clip manager with domain: "+ domain, "for user_id: " ,user_id)
            print("File mode: ", training)
        else: 
            print("No domain provided.")
        print("---------------")
        print("---------------")
        self.domain = domain
        self.user_id = user_id
        self._upload_workers = multiprocessing.Pool(workers)
        self._pending_upload_results = []


        ################################################
        # Esto esta a False para que no consuma memoria #
        # Para entrenar modelos, poner a True           #
        ################################################
        self.file_mode = training
        # self.file_mode = True
        ################################################
        ################################################

        self._clips = {}

        # Load clips from database and disk. Filter by domain if provided.
        from human_feedback_api import Clip
        
        if self.domain:
            self.clips_queryset = Clip.objects.filter(environment_id=self.env_id, domain=self.domain)
        else:
            self.clips_queryset = Clip.objects.filter(environment_id=self.env_id, domain__isnull=True)
        print("This is the queryset for this manager: ", self.clips_queryset)
        if self.file_mode:
            for clip in self.clips_queryset:
                clip_id = clip.clip_tracking_id
                try:
                    self._clips[clip_id] = pickle.load(open(self._pickle_path(clip_id), 'rb'))
                except FileNotFoundError:
                    pass
                except Exception:
                    print("Exception occurred when loading clip %s" % clip_id)
                    if input("Do you want to erase this clip? (y/n)\n> ").lower().startswith('y'):
                        print("Erasing clip from disk...")
                        os.remove(self._pickle_path(clip_id))
                        print("Erasing clip and all related data from database...")
                        clip.delete()
                        print("Warning: There's a chance that this simply invalidates the sort-tree. Check the human feedback api /tree/experiment_name for any experiments involving this clip.")
                        from human_feedback_api import Comparison
                        Comparison.objects.filter(left_clip=clip).delete()
                        Comparison.objects.filter(right_clip=clip).delete()
                        print("Invalid data deleted.\nMoving on...")
                    else:
                        raise
        if self.file_mode:
            self._max_clip_id = max(self._clips.keys()) if self._clips else 0
        else:
            self._max_clip_id = max((clip.clip_tracking_id for clip in self.clips_queryset), default=0)

        self._sorted_clips = []  # List of lists of clip_ids
        self.sort_clips()        

        # Report
        if self.total_number_of_clips < 1:
            print("Starting fresh!")
        else:
            print(f"Found {self.total_number_of_clips} old clips for this environment{' and domain' if self.domain else ''}! ({self.number_of_sorted_clips} sorted)")

    def free_memory(self):
        """Clear loaded clips from memory."""
        self._clips.clear() 
        print("Cleared loaded clips from memory.")


    def create_new_sort_tree_from_existing_clips(self):
        from human_feedback_api import Clip
        # Assume that the best seed clip is the one with the lowest tracking id
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        print("create_new_sort_tree_from_existing_clips")
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        if self.file_mode:
            seed_id = min(self._clips.keys())
            seed_clip = Clip.objects.get(environment_id=self.env_id, domain=self.domain, clip_tracking_id=seed_id)
        else:
            seed_clip = self.clips_queryset.order_by('clip_tracking_id').first()

        self._create_sort_tree(seed_clip)
        if self.file_mode:
            for clip_id in self._clips:
                if clip_id != seed_id:
                    clip = Clip.objects.get(environment_id=self.env_id, domain=self.domain, clip_tracking_id=clip_id)
                    print("\tASSIGNING CLIP TO SORT TREE!!!")
                    self._assign_clip_to_sort_tree(clip)
                    print("\tASSIGNINED CLIP TO SORT TREE!!!")
        else: 
            for clip_id in self._clips:
                print("for clip_id in self._clips:: ", clip_id)

            for clip in self.clips_queryset:
                print("clip in self.clips_queryset ", clip)

                if clip.clip_tracking_id != seed_clip.clip_tracking_id:
                    print("\tASSIGNING CLIP TO SORT TREE!!!")
                    self._assign_clip_to_sort_tree(clip)
                    print("\tASSIGNED CLIP TO SORT TREE!!!")

    def _create_sort_tree(self, seed_clip):
        from human_feedback_api import SortTree
        tree = SortTree(
            experiment_name=self.experiment_name,
            domain = self.domain,
            user_id = self.user_id,
            is_red=False,
        )
        tree.save()
        tree.bound_clips.add(seed_clip)

    def _assign_clip_to_sort_tree(self, clip):
        from human_feedback_api import SortTree
        try:
            print("_assign_clip_to_sort_tree::: ")
            if self.user_id:
                root = SortTree.objects.get(experiment_name=self.experiment_name, domain=self.domain, user_id=self.user_id, parent=None)
            else:
                root = SortTree.objects.get(experiment_name=self.experiment_name, domain=self.domain, parent=None)
            print("_assign_clip_to_sort_tree:: root: ", root)
            root.pending_clips.add(clip)
        except SortTree.DoesNotExist:
            self._create_sort_tree(clip)

    def _add_to_database(self, clip_id, source="", actions=""):
        from human_feedback_api import Clip
        clip = Clip(
            environment_id=self.env_id,
            clip_tracking_id=clip_id,
            domain=self.domain,
            media_url=self._gcs_url(clip_id),
            source=source,
            actions= actions,
        )
        clip.save()
        self._assign_clip_to_sort_tree(clip)

    def add(self, new_clip, *, source="", sync=False, actions=""):
        print("trying to add: ", actions)
        clip_id = self._max_clip_id + 1
        self._max_clip_id = clip_id
        self._clips[clip_id] = new_clip
        # Get the frames
        frames = [self.env.render_full_obs(x) for x in new_clip["human_obs"]]  # We do this here because render_full_obs is dangerous to pass to a subprocess
        # Write the clip to disk and upload
        if sync:
            uploaded_clip_id, _, _ = _write_and_upload_video(
                new_clip, frames, clip_id, source, self.env.fps, self._gcs_path(clip_id), self._video_path(clip_id), self._pickle_path(clip_id), actions)
            self._add_to_database(uploaded_clip_id, source, actions=actions)
        else:  # async
            self._pending_upload_results.append(self._upload_workers.apply_async(_write_and_upload_video, (
                new_clip, frames, clip_id, source, self.env.fps, self._gcs_path(clip_id), self._video_path(clip_id), self._pickle_path(clip_id), actions)))
        # Avoid memory leaks!
        self._check_pending_uploads()

    def _check_pending_uploads(self):
        # Check old pending results to see if we can clear memory and add them to the database. Also reveals errors.
        for pending_result in self._pending_upload_results:
            if pending_result.ready():
                uploaded_clip_id, uploaded_clip_source, actions = pending_result.get(timeout=60)
                print(" ---- ", actions)
                self._add_to_database(uploaded_clip_id, uploaded_clip_source, actions=actions)
        self._pending_upload_results = [r for r in self._pending_upload_results if not r.ready()]

    @property
    def total_number_of_clips(self):
        if(self.clips_queryset and not self.file_mode):
            return self.clips_queryset.count() 
        return len(self._clips)

    @property
    def number_of_sorted_clips(self):
        return sum([len(self._sorted_clips[i]) for i in range(len(self._sorted_clips))])

    @property
    def maximum_ordinal(self):
        return len(self._sorted_clips) - 1

    def sort_clips(self, wait_until_database_fully_sorted=False):
        from human_feedback_api import SortTree
        print("SORT CLIPS!!! : ", wait_until_database_fully_sorted)
        if wait_until_database_fully_sorted:
            print("Waiting until all clips in the database are sorted...")
            while self._pending_upload_results or SortTree.objects.filter(experiment_name=self.experiment_name, domain=self.domain, user_id=self.user_id, pending_clips__isnull=False):
                self._check_pending_uploads()
                print("Finished cheking pending uploads")
                sleep(10)
                print("_pending_upload_results::: ", self._pending_upload_results)
                print("_pending_upload_results::: ", bool(self._pending_upload_results))
                print("SortTree.objects.filter(experiment_name=self.experiment_name, domain=self.domain, user_id=self.user_id, pending_clips__isnull=False)::: ", SortTree.objects.filter(experiment_name=self.experiment_name, domain=self.domain, user_id=self.user_id, pending_clips__isnull=False))
                print("bool: SortTree.objects.filter(experiment_name=self.experiment_name, domain=self.domain, user_id=self.user_id, pending_clips__isnull=False)::: ", bool(SortTree.objects.filter(experiment_name=self.experiment_name, domain=self.domain, user_id=self.user_id, pending_clips__isnull=False)))
            print("Okay! The database seems to be sorted!")
        sorted_clips = []
        try:
            if self.user_id:
                print("\tSe ha pasado un user_ID ", self.user_id, ". Filtrando por user ID.")
                node = _tree_minimum(SortTree.objects.get(experiment_name=self.experiment_name, domain=self.domain, user_id=self.user_id ,parent=None))
                print("NODE: ", node)
            else: 
                print("\tNo se ha pasado ningun User_id.")
                node = _tree_minimum(SortTree.objects.get(experiment_name=self.experiment_name, domain=self.domain ,parent=None))
                print("NODE: ", node)

            # node = _tree_minimum(SortTree.objects.get(experiment_name=self.experiment_name, domain=self.domain, user_id=self.user_id ,parent=None))
            while node:
                sorted_clips.append([x.clip_tracking_id for x in node.bound_clips.all()])
                node = _tree_successor(node)
        except SortTree.DoesNotExist:
            pass  # Root hasn't been created.
        self._sorted_clips = sorted_clips

    def get_sorted_clips(self, *, batch_size=None):
        clip_ids_with_ordinals = []
        for ordinal in range(len(self._sorted_clips)):
            clip_ids_with_ordinals += [{'id': clip_id, 'ord': ordinal} for clip_id in self._sorted_clips[ordinal]]
        sample = np.random.choice(clip_ids_with_ordinals, batch_size, replace=False) if batch_size else clip_ids_with_ordinals
        return [item['id'] for item in sample], [self._clips[item['id']] for item in sample], [item['ord'] for item in sample]

    def _video_filename(self, clip_id):
        return "%s-%s-%s.mp4" % (self.experiment_name, self.domain, clip_id)

    def _video_path(self, clip_id):
        # Check if the platform is Windows.
        if os.name == 'nt':
            # Use the USERPROFILE environment variable for the home directory
            home_dir = os.environ.get('USERPROFILE', 'C:/Users')
        else:
            # For non-Windows platforms, use the HOME environment variable
            home_dir = os.environ.get('HOME', '/home')

        # Define the possible folder names
        possible_folders = ['Documents', 'Documentos']

        # Iterate over possible folder names
        for folder_name in possible_folders:
            folder = os.path.join(home_dir, folder_name, 'rl_teacher_media')
            # Check if the folder exists
            if os.path.exists(folder):
                break  # Exit loop if the folder exists

        # If no existing folder found, use the first one
        else:
            folder = os.path.join(home_dir, possible_folders[0], 'rl_teacher_media')

        tmp_folder = os.path.join('C:', 'tmp', 'rl_teacher_media')  # Assuming 'C:' is your system drive
        return os.path.join(folder, self._video_filename(clip_id))

    def _pickle_path(self, clip_id):
        return os.path.join('clips', '%s-%s-%s.clip' % (self.env_id, self.domain, clip_id))

    def _gcs_path(self, clip_id):
        return self._video_path(clip_id)
        # return os.path.join(self.gcs_bucket, self._video_filename(clip_id))

    def _gcs_url(self, clip_id):
        # return "http://127.0.0.1:5000/%s" % (self._video_filename(clip_id))
        return "http://158.42.185.67:5000/%s" % (self._video_filename(clip_id))
        # return self._video_path(clip_id)
        # return "https://storage.googleapis.com/%s/%s" % (self.gcs_bucket.lstrip("gs://"), self._video_filename(clip_id))
