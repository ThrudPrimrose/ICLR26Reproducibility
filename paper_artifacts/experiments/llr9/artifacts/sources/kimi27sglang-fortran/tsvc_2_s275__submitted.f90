subroutine tsvc_2_s275_fp64(aa, bb, cc, LEN_2D) bind(c, name='tsvc_2_s275_fp64')
  use iso_c_binding, only: c_double, c_int64_t
  use omp_lib
  implicit none
  real(c_double), intent(inout) :: aa(0:*)
  real(c_double), intent(in) :: bb(0:*), cc(0:*)
  integer(c_int64_t), value, intent(in) :: LEN_2D
  integer(c_int64_t) :: i, j, tid, nthreads, istart0, iend0, istart, iend

  if (LEN_2D < 128) then
    do j = 1, LEN_2D - 1
      do i = 0, LEN_2D - 1
        if (aa(i) > 0.0_c_double) then
          aa(j * LEN_2D + i) = aa((j - 1) * LEN_2D + i) + bb(j * LEN_2D + i) * cc(j * LEN_2D + i)
        end if
      end do
    end do
    return
  end if

  !$omp parallel private(i, j, tid, nthreads, istart0, iend0, istart, iend)
  nthreads = omp_get_num_threads()
  tid = omp_get_thread_num()
  istart0 = (tid * LEN_2D) / nthreads
  iend0 = ((tid + 1) * LEN_2D) / nthreads - 1
  istart = ((istart0 + 7) / 8) * 8
  iend = istart + ((iend0 - istart + 1) / 8) * 8 - 1
  do j = 1, LEN_2D - 1
    do i = istart0, istart - 1
      if (aa(i) > 0.0_c_double) then
        aa(j * LEN_2D + i) = aa((j - 1) * LEN_2D + i) + bb(j * LEN_2D + i) * cc(j * LEN_2D + i)
      end if
    end do
    !$omp simd
    do i = istart, iend
      if (aa(i) > 0.0_c_double) then
        aa(j * LEN_2D + i) = aa((j - 1) * LEN_2D + i) + bb(j * LEN_2D + i) * cc(j * LEN_2D + i)
      end if
    end do
    !$omp end simd
    do i = iend + 1, iend0
      if (aa(i) > 0.0_c_double) then
        aa(j * LEN_2D + i) = aa((j - 1) * LEN_2D + i) + bb(j * LEN_2D + i) * cc(j * LEN_2D + i)
      end if
    end do
  end do
  !$omp end parallel
end subroutine tsvc_2_s275_fp64
