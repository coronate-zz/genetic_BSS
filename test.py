        all_list = list()
        for i in FLEET.fleet:
            car = FLEET.fleet[i]
            print(len(car.subsystem_list))
            all_list.extend(car.subsystem_list)