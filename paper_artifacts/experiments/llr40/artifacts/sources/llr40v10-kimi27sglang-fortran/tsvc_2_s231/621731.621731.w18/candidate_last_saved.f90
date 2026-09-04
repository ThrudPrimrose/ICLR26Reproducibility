! Optimized Fortran implementation of TSVC s231
subroutine tsvc_2_s231_fp64(aa, bb, LEN_2D) bind(C, name='tsvc_2_s231_fp64')
  use iso_c_binding, only: c_double, c_int64_t
  use omp_lib, only: omp_get_thread_num, omp_get_num_threads
  implicit none
  integer(c_int64_t), intent(in), value :: LEN_2D
  real(c_double), intent(inout) :: aa(LEN_2D, LEN_2D)
  real(c_double), intent(in)    :: bb(LEN_2D, LEN_2D)
  integer(c_int64_t) :: i, j, i0, i1
  integer :: nt, tid, rem
  integer(c_int64_t) :: chunk

  !$omp parallel default(none) shared(aa, bb, LEN_2D) private(i, j, i0, i1, nt, tid, chunk, rem)
  nt = omp_get_num_threads()
  tid = omp_get_thread_num()
  chunk = LEN_2D / int(nt, kind=c_int64_t)
  rem = int(mod(LEN_2D, int(nt, kind=c_int64_t)))
  if (tid < rem) then
    i0 = int(tid, kind=c_int64_t) * (chunk + 1_c_int64_t) + 1_c_int64_t
    i1 = i0 + chunk
  else
    i0 = int(tid, kind=c_int64_t) * chunk + int(rem, kind=c_int64_t) + 1_c_int64_t
    i1 = i0 + chunk - 1_c_int64_t
  end if

  do j = 2_c_int64_t, LEN_2D
    do i = i0, i1
      aa(i, j) = aa(i, j - 1_c_int64_t) + bb(i, j)
    end do
  end do
  !$omp end parallel
end subroutine tsvc_2_s231_fp64
