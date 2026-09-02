subroutine tsvc_2_s1244_fp64(pa, pb, pc, pd, n) bind(C, name = "tsvc_2_s1244_fp64")
  use, intrinsic :: iso_c_binding, only: c_ptr, c_int64_t, c_f_pointer
  implicit none
  type(c_ptr), value :: pa, pb, pc, pd
  integer(c_int64_t), value :: n
  real(kind=8), pointer, contiguous :: a(:), b(:), c(:), d(:)
  integer(c_int64_t) :: i
  integer :: nshape

  if (n < 2) return
  nshape = int(n)
  call c_f_pointer(pa, a, [nshape])
  call c_f_pointer(pb, b, [nshape])
  call c_f_pointer(pc, c, [nshape])
  call c_f_pointer(pd, d, [nshape])

  ! Reference (sequential):
  !   for i = 0..n-2:  a[i] = b[i] + c[i]*c[i] + b[i]*b[i] + c[i]
  !                    d[i] = a[i] + a[i+1]      (a[i+1] still the OLD value)
  ! d(i) uses the ORIGINAL a(i+1), so compute d first while a is pristine,
  ! then update a. Both loops are fully independent.
  !$omp parallel do default(none) shared(a,b,c,d,n) schedule(static)
  do i = 1, n - 1
    d(i) = b(i) + c(i)*c(i) + b(i)*b(i) + c(i) + a(i+1)
  end do
  !$omp parallel do default(none) shared(a,b,c,d,n) schedule(static)
  do i = 1, n - 1
    a(i) = b(i) + c(i)*c(i) + b(i)*b(i) + c(i)
  end do
end subroutine tsvc_2_s1244_fp64
