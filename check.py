from statistics import median

all_sums  = []
all_times = []


while True:
    sums  = input().split()
    if len(sums) == 1:
        break

    times = input().split()

    all_sums.append(float(sums[-1]))
    all_times.append(float(times[-1]))


print("Sums eps: ", max(all_sums) - min(all_sums))
print("Average sum: ", median(all_sums))

print("time eps: ", max(all_times) - min(all_times))
print("Average time: ", median(all_sums))
