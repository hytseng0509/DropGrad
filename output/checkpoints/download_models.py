from subprocess import call

# current available models:
# 1/5 shot, MAML, gaussian DropGrad (rate = 0.1)

filename = '5s_MAML_gaussian_01' + '.tar.gz'

call('wget http://vllab.ucmerced.edu/ym41608/projects/DropGrad/checkpoints/' + filename, shell=True)
call('tar -zxf ' + filename, shell=True)
call('rm ' + filename, shell=True)
