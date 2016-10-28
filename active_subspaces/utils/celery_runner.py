from celery import Celery
import marshal, types

celery = Celery('active-subspace')
#celery.config_from_object('../../celeryconfig')
celery.config_from_object('celeryconfig')


# http://stackoverflow.com/questions/1253528/is-there-an-easy-way-to-pickle-a-python-function-or-otherwise-serialize-its-cod/1253813#1253813

@celery.task()
def celery_runner(x, marshal_func):
	code = marshal.loads(marshal_func)
	func = types.FunctionType(code, globals(), "func")
	return func(x)

