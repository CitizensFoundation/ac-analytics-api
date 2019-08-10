def get_memory_usage():
  """
  Get node total memory and memory usage
  """
  with open('/proc/meminfo', 'r') as mem:
      ret = {}
      tmp = 0
      for i in mem:
          sline = i.split()
          if str(sline[0]) == 'MemTotal:':
              ret['total'] = int(sline[1])
          elif str(sline[0]) in ('MemFree:', 'Buffers:', 'Cached:'):
              tmp += int(sline[1])
      ret['free'] = tmp
      ret['used'] = int(ret['total']) - int(ret['free'])
  return ret