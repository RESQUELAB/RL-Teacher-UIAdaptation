from django.conf.urls import include, url

from django.contrib import admin
admin.autodiscover()

import human_feedback_api.views

# Examples:
# url(r'^$', 'human_comparison_site.views.home', name='home'),
# url(r'^blog/', include('blog.urls')),

urlpatterns = [
    url(r'^$', human_feedback_api.views.index, name='index'),
    url(r'^experiments/(.*)/list$', human_feedback_api.views.list_comparisons, name='list'),
    url(r'^comparisons/(.*)$', human_feedback_api.views.show_comparison, name='show_comparison'),
    url(r'^experiments/(.*)/ajax_response$', human_feedback_api.views.ajax_response, name='ajax_response'),
    url(r'^experiments/(.*)$', human_feedback_api.views.respond, name='responses'),
    url(r'^admin/', include(admin.site.urls)),
    url(r'^tree/(.*)$', human_feedback_api.views.tree, name='tree_viewer'),
    url(r'^clips/(.*)$', human_feedback_api.views.all_clips, name='all_clips'),
    url(r'^register/$', human_feedback_api.views.register, name='register'),
    url(r'^register_web/$', human_feedback_api.views.user_register, name='register_web'),
    url(r'^login/$', human_feedback_api.views.login_user, name='login'),
    url(r'^login_web/$', human_feedback_api.views.user_login, name='login_web'),
    url(r'^logout/$', human_feedback_api.views.logout_user, name='logout'),
    url(r'^api/log_training_completion/$', human_feedback_api.views.log_training_completion, name='log_training_completion'),
    ]
