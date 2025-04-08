from django.core.exceptions import ValidationError
from django.db import models
from django.utils.translation import ugettext_lazy as _
from django.contrib.auth.models import User
from django.contrib.postgres.fields import JSONField
from django.db.models.signals import post_save
from django.dispatch import receiver


RESPONSE_KIND_TO_RESPONSES_OPTIONS = {'left_or_right': ['left', 'right', 'tie', 'abstain']}

def validate_inclusion_of_response_kind(value):
    kinds = RESPONSE_KIND_TO_RESPONSES_OPTIONS.keys()
    if value not in kinds:
        raise ValidationError(_('%(value)s is not included in %(kinds)s'), params={'value': value, 'kinds': kinds}, )

class Clip(models.Model):
    created_at = models.DateTimeField('date created', auto_now_add=True, db_index=True)
    updated_at = models.DateTimeField(auto_now=True, db_index=True)

    media_url = models.TextField('media url', db_index=True)
    environment_id = models.TextField('environment id', db_index=True)
    clip_tracking_id = models.IntegerField('clip tracking id', db_index=True)
    domain = models.CharField(max_length=255, db_index=True, blank=True, null=None)
    
    source = models.TextField('note of where the clip came from', default="", blank=True)
    
    actions = models.TextField('note of where the actions made in this clip', default="", blank=True)

    def __str__(self):
        return f"Clip {self.clip_tracking_id} from domain {self.domain or 'No Domain'}"

class Comparison(models.Model):
    created_at = models.DateTimeField('date created', auto_now_add=True, db_index=True)
    updated_at = models.DateTimeField(auto_now=True, db_index=True)

    left_clip = models.ForeignKey(Clip, db_index=True, related_name="compared_on_the_left")
    right_clip = models.ForeignKey(Clip, db_index=True, related_name="compared_on_the_right")

    shown_to_tasker_at = models.DateTimeField('time shown to tasker', db_index=True, blank=True, null=True)
    responded_at = models.DateTimeField('time response received', db_index=True, blank=True, null=True)
    response_kind = models.TextField('the response from the tasker', db_index=True,
                                     validators=[validate_inclusion_of_response_kind])
    response = models.TextField('the response from the tasker', db_index=True, blank=True, null=True)
    experiment_name = models.TextField('name of experiment')

    priority = models.FloatField('site will display higher priority items first', db_index=True)
    note = models.TextField('note to be displayed along with the query', default="", blank=True)

    # The Binary Search/Sort Tree that this comparison belongs to. Only used for new-style experiments.
    tree_node = models.ForeignKey('SortTree', null=True, blank=True, default=None)
    # Whether this comparison is related to a pending clip for said node. Helper used for new-style experiments.
    relevant_to_pending_clip = models.BooleanField(default=False)

    # Relation to the user who made the comparison
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='compares', default=None)    
    
    def __str__(self):
        return "Comparison {} ({} vs {})".format(self.id, self.left_clip, self.right_clip)

    # Validation
    def full_clean(self, exclude=None, validate_unique=True):
        super(Comparison, self).full_clean(exclude=exclude, validate_unique=validate_unique)
        self.validate_inclusion_of_response()

    @property
    def response_options(self):
        try:
            return RESPONSE_KIND_TO_RESPONSES_OPTIONS[self.response_kind]
        except KeyError:
            raise KeyError("{} is not a valid response_kind. Valid response_kinds are {}".format(
                self.response_kind, RESPONSE_KIND_TO_RESPONSES_OPTIONS.keys()))

    def validate_inclusion_of_response(self):
        # This can't be a normal validator because it depends on a value
        if self.response is not None and self.response not in self.response_options:
            raise ValidationError(
                _('%(value)s is not included in %(options)s'),
                params={'value': self.response, 'options': self.response_options}, )

class SortTree(models.Model):
    """ Extends a red-black tree to handle async clip sorting with equivalence. """

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='sort_trees')    
    parent = models.ForeignKey('self', null=True, related_name='+')
    left = models.ForeignKey('self', null=True, related_name='+')
    right = models.ForeignKey('self', null=True, related_name='+')

    pending_clips = models.ManyToManyField(Clip, related_name='pending_sort_locations')
    bound_clips = models.ManyToManyField(Clip, related_name='tree_bindings')

    experiment_name = models.TextField('name of experiment')

    domain = models.CharField(max_length=255, db_index=True, blank=True, null=None)

    is_red = models.BooleanField()  # Used for red-black autobalancing

    def __str__(self):
        return "Node {}".format(self.id)

    # I could theoretically do these with a setter decorator,
    # but I want to be able to manipulate them directly without autosaving if needed.
    def make_red(self):
        self.is_red = True
        self.save()

    def make_black(self):
        self.is_red = False
        self.save()

    def set_left(self, x):
        self.left = x
        if x:
            x.user = self.user
            x.parent = self
            x.save()
        self.save()

    def set_right(self, x):
        self.right = x
        if x:
            x.user = self.user
            x.parent = self
            x.save()
        self.save()


class TrainingCompletion(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    domain = models.CharField(max_length=255)
    experiment = models.CharField(max_length=255)
    environment = models.CharField(max_length=255)
    completed_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ('user', 'domain', 'experiment', 'environment')
        verbose_name = 'Training Completion'
        verbose_name_plural = 'Training Completions'

    def __str__(self):
        return f"{self.user} - {self.domain} - {self.experiment} - {self.environment}"

class Profile(models.Model):
    ''' This class extends the user model information '''
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    group = models.IntegerField()
    experiment = models.TextField(max_length=200, blank = True)
    birth = models.TextField(max_length=200, blank = True)
    gender = models.TextField(max_length=200, blank = True)
    occupations = models.TextField(max_length=200, blank = True)
    field = models.TextField(max_length=200, blank = True)
    background = models.TextField(max_length=200, blank = True)
    usage_frequency_pc = models.TextField(max_length=200, blank = True)
    usage_frequency_smartphone = models.TextField(max_length=200, blank = True)
    usage_frequency_tablet = models.TextField(max_length=200, blank = True)
    usage_frequency_console = models.TextField(max_length=200, blank = True)


class SurveyResponse(models.Model):
    """Stores user survey responses"""
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    group = models.IntegerField()
    domain = models.TextField()
    experiment = models.TextField()
    responses = JSONField()  
    timestamp = models.DateTimeField(auto_now_add=True)


@receiver(post_save, sender=User)
def create_user_profile(sender, instance, created, **kwargs):
    if created:
        Profile.objects.create(user=instance)

@receiver(post_save, sender=User)
def save_user_profile(sender, instance, **kwargs):
    instance.profile.save()