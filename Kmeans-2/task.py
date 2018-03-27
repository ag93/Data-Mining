import kMeans as kmeans

print("##################")
print("Task 1: Run k-means on AT&T 100 images, set K=10:")

k =10
C_matrix, accuracy = kmeans.predict("ATNTFaceImages100.txt", k)
with open('confusion_matrix1.txt', 'w') as f:
    f.write(str(C_matrix))
f.close()
print("Accuracy:",accuracy)

print("\n##################")
print("Task 2: Run k-means on AT&T 400 images, set K=40")

k = 40
C_matrix, accuracy = kmeans.predict("ATNTFaceImages400.txt", k)
with open('confusion_matrix2.txt', 'w') as f:
    f.write(str(C_matrix))
f.close()
print("Accuracy:",accuracy)

print("\n##################")
print("Task 3: Run k-means on Hand-written-letters data, set K=26")

k = 26
C_matrix, accuracy = kmeans.predict("HandWrittenLetters.txt", k)
with open('confusion_matrix3.txt', 'w') as f:
    f.write(str(C_matrix))
f.close()
print("Accuracy:",accuracy)
