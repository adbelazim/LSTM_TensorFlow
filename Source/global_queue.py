subjects_list = []
cases_list = []
orders_list = []

def push_actual_subject(subject,subjects_list):

	subjects_list.append(subject)
	#print("lista sujeto actual", subjects_list)
	return subjects_list

def get_actual_subject(subjects_list):

	actual_subject = subjects_list[0]
	#print("cuando obtengo el tamanio es ", subjects_list)
	#print("Obtuve al sujeto", actual_subject)
	return actual_subject

def pop_actual_subject(subjects_list):
	subjects_list.pop()

#############case managment#########

def push_actual_case(case,cases_list):

	cases_list.append(case)
	#print("lista caso actual", cases_list)
	return cases_list

def get_actual_case(cases_list):

	actual_case = cases_list[0]
	#print("cuando obtengo el tamanio es ", cases_list)
	#print("Obtuve al caso", actual_case)
	return actual_case

def pop_actual_case(cases_list):
	cases_list.pop()

#############order managment#########

def push_actual_order(order,orders_list):

	orders_list.append(order)
	#print("lista order actual", orders_list)
	return orders_list

def get_actual_order(orders_list):

	actual_order = orders_list[0]
	#print("cuando obtengo el tamanio es ", orders_list)
	#print("Obtuve al order", actual_order)
	return actual_order

def pop_actual_order(orders_list):
	orders_list.pop()



