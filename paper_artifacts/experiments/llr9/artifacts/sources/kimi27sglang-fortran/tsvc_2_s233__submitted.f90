module tsvc_2_s233_mod
  use, intrinsic :: iso_c_binding, only: c_double, c_int64_t
  use :: omp_lib, only: omp_get_thread_num, omp_get_num_threads
  implicit none
contains
  subroutine tsvc_2_s233_fp64(aa, bb, cc, len_2d) bind(c, name='tsvc_2_s233_fp64')
    implicit none
    integer(c_int64_t), value, intent(in) :: len_2d
    real(c_double), intent(inout) :: aa(len_2d, len_2d)
    real(c_double), intent(inout) :: bb(len_2d, len_2d)
    real(c_double), intent(in)    :: cc(len_2d, len_2d)
    integer(c_int64_t) :: i, j, n, tid, nt, ib, ie, chunk, rem

    n = len_2d - 8_c_int64_t

    !$omp parallel default(none) shared(aa, bb, cc, len_2d, n) private(i, j, tid, nt, ib, ie, chunk, rem)

      tid = omp_get_thread_num()
      nt  = omp_get_num_threads()
      chunk = n / nt
      rem   = mod(n, nt)
      if (tid < rem) then
        chunk = chunk + 1_c_int64_t
        ib = 9_c_int64_t + tid * chunk
      else
        ib = 9_c_int64_t + rem * (chunk + 1_c_int64_t) + (tid - rem) * chunk
      end if
      ie = ib + chunk - 1_c_int64_t

      do j = 9_c_int64_t, len_2d
        do i = ib, ie
          aa(i, j) = aa(i, j - 1) + cc(i, j)
        end do
      end do

      !$omp do schedule(static)
      do j = 9_c_int64_t, len_2d
        do i = 9_c_int64_t, len_2d
          bb(i, j) = bb(i - 1, j) + cc(i, j)
        end do
      end do
      !$omp end do

    !$omp end parallel

  end subroutine tsvc_2_s233_fp64
end module tsvc_2_s233_mod
