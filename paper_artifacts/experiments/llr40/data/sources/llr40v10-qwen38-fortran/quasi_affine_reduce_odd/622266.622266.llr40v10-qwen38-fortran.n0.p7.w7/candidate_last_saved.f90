subroutine quasi_affine_reduce_odd_fp64(a_ptr, out_ptr, len_1d) bind(C, name="quasi_affine_reduce_odd_fp64")
  use, intrinsic :: iso_c_binding
  use, intrinsic :: omp_lib
  implicit none
  type(c_ptr), value, intent(in) :: a_ptr, out_ptr
  integer(c_int64_t), value, intent(in) :: len_1d
  real(c_double), pointer :: A(:)
  real(c_double), pointer :: O(:)
  real(c_double) :: acc, s, a1,a2,a3,a4,a5,a6,a7,a8
  integer(kind=8) :: i, nodd, tmax
  real(c_double) :: t0, t1

  call c_f_pointer(a_ptr, A, [len_1d])
  call c_f_pointer(out_ptr, O, [1])
  nodd = len_1d / 2
  tmax = omp_get_max_threads()

  t0 = omp_get_wtime()
  s = 0.0d0
  do i = 1, len_1d
     s = s + A(i)
  end do
  t1 = omp_get_wtime()
  print *, "E1_1thr_all ns=", (t1-t0)*1e9, " val=", s

  a1=0.0d0;a2=0.0d0;a3=0.0d0;a4=0.0d0;a5=0.0d0;a6=0.0d0;a7=0.0d0;a8=0.0d0
  do i = 2, 2*(nodd-8), 16
     a1 = a1 + A(i)
     a2 = a2 + A(i+2)
     a3 = a3 + A(i+4)
     a4 = a4 + A(i+6)
     a5 = a5 + A(i+8)
     a6 = a6 + A(i+10)
     a7 = a7 + A(i+12)
     a8 = a8 + A(i+14)
  end do
  do i = 2*(nodd-8)+16, 2*nodd, 2
     acc = A(i)
  end do
  acc = a1+a2+a3+a4+a5+a6+a7+a8+acc
  t1 = omp_get_wtime()
  print *, "E2_1thr_8acc ns=", (t1-t0)*1e9, " val=", acc

  acc = 0.0d0
  !$omp parallel do reduction(+:acc)
  do i = 2, len_1d, 2
     acc = acc + A(i)
  end do
  !$omp end parallel do
  t1 = omp_get_wtime()
  print *, "E3_omp_n1acc nthreads=", tmax, " ns=", (t1-t0)*1e9, " val=", acc

  acc = 0.0d0
  !$omp parallel do reduction(+:acc)
  do i = 2, 2*(nodd-8), 16
     a1 = A(i); a2 = A(i+2); a3 = A(i+4); a4 = A(i+6)
     a5 = A(i+8); a6 = A(i+10); a7 = A(i+12); a8 = A(i+14)
     acc = acc + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8
  end do
  !$omp end parallel do
  do i = 2*(nodd-8)+16, 2*nodd, 2
     acc = acc + A(i)
  end do
  t1 = omp_get_wtime()
  print *, "E4_omp_8acc ns=", (t1-t0)*1e9, " val=", acc

  acc = 0.0d0
  !$omp parallel do reduction(+:acc)
  do i = 2, len_1d, 2
     acc = acc + A(i)
  end do
  !$omp end parallel do
  t1 = omp_get_wtime()
  print *, "E5_omp_repeat ns=", (t1-t0)*1e9, " val=", acc

  s = 0.0d0
  !$omp parallel do reduction(+:s)
  do i = 1, len_1d
     s = s + A(i)
  end do
  !$omp end parallel do
  t1 = omp_get_wtime()
  print *, "E6_omp_all ns=", (t1-t0)*1e9, " val=", s

  print *, "nprocs=", omp_get_num_procs(), " numthreads=", tmax
  O(1) = 0.0d0
end subroutine
