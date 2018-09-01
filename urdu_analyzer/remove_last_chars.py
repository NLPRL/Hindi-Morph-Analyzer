import re

def remove_recurring_from_last(s):
	s = s[::-1] # reverse the string
	l = len(s)

	cnt = 0 # store the count of repeating characters to remove
	for i in range(l):
		if (s[i] == s[i+1]):
			cnt += 1
		else:
			break

	print(cnt)
	s = s.replace(s[0],'',cnt)

	result = s[::-1]

	return result

str = 'aaaa'

print(remove_recurring_from_last(str))