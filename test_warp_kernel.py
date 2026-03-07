import warp as wp
wp.init()

@wp.kernel
def test(a: wp.array(dtype=float)):
    i = wp.tid()
    a[i] = float(i)

arr = wp.zeros(10, dtype=float, device="cuda:0")
wp.launch(test, dim=10, inputs=[arr], device="cuda:0")
print("Kernel OK:", arr.numpy())
