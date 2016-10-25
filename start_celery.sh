#! /bin/bash
celery -A active_subspaces.celery worker
