subroutine tsvc_2_s3112_fp64(a, b, len_1d) bind(C, name='tsvc_2_s3112_fp64')
  use, intrinsic :: omp_lib
  implicit none
  real(kind=8), intent(in)    :: a(*)
  real(kind=8), intent(out)   :: b(*)
  integer(kind=8), value      :: len_1d
  integer(kind=8) :: i
  real(kind=8) :: sum
  real(kind=8) :: amax, bsum
  integer :: mt
  character(len=64) :: buf

  mt = omp_get_max_threads()
  write(buf, '(a,i0)') 'JUDGE_MAX_THREADS=', mt
  print '("JUDGE_MAX_THREADS=", i0)', mt
  print '("N=", i0)', len_1d
  amax = 0.0d0
  bsum = 0.0d0
  do i = 1, len_1d
     if (abs(a(i)) > amax) amax = abs(a(i))
     bsum = bsum + a(i)
  end do
  print '("AMAX=", es15.6)', amax
  print '("ABSSUM=", es15.6)', abs(bsum)
  sum = 0.0d0
  do i = 1, len_1d
     sum = sum + a(i)
     b(i) = sum
  end do
  print '("DONE")'
end subroutine tsvc_2_s3112_fp64
