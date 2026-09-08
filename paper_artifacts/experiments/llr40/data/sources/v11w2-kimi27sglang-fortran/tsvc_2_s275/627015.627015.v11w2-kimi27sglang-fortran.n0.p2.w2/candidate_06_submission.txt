subroutine tsvc_2_s275_fp64(aa, bb, cc, LEN_2D) bind(c, name="tsvc_2_s275_fp64")
  use iso_c_binding, only: c_double, c_int64_t
  use omp_lib
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_2D
  real(c_double), intent(inout) :: aa(LEN_2D, LEN_2D)
  real(c_double), intent(in) :: bb(LEN_2D, LEN_2D), cc(LEN_2D, LEN_2D)

  integer(c_int64_t) :: i, j, istart, iend, chunk
  integer :: tid, nthreads
  integer(c_int64_t), parameter :: VEC = 8_c_int64_t
  logical :: mask(LEN_2D)

  if (LEN_2D <= 512_c_int64_t) then
    do j = 2, LEN_2D
      !$omp simd simdlen(8)
      do i = 1, LEN_2D
        if (aa(i, 1) > 0.0_c_double) then
          aa(i, j) = aa(i, j - 1) + bb(i, j) * cc(i, j)
        end if
      end do
    end do
    return
  end if

  do i = 1, LEN_2D
    mask(i) = aa(i, 1) > 0.0_c_double
  end do

  !$omp parallel default(none) private(i, j, tid, chunk, istart, iend) shared(aa, bb, cc, mask, LEN_2D, nthreads)
  tid = omp_get_thread_num()
  nthreads = omp_get_num_threads()
  chunk = ((LEN_2D + int(nthreads, c_int64_t) * VEC - 1_c_int64_t) / (int(nthreads, c_int64_t) * VEC)) * VEC
  istart = int(tid, c_int64_t) * chunk + 1_c_int64_t
  iend = min(istart + chunk - 1_c_int64_t, LEN_2D)

  if (istart <= LEN_2D) then
    do j = 2, LEN_2D
      !$omp simd simdlen(8)
      do i = istart, iend
        if (mask(i)) then
          aa(i, j) = aa(i, j - 1) + bb(i, j) * cc(i, j)
        end if
      end do
    end do
  end if
  !$omp end parallel
end subroutine tsvc_2_s275_fp64
