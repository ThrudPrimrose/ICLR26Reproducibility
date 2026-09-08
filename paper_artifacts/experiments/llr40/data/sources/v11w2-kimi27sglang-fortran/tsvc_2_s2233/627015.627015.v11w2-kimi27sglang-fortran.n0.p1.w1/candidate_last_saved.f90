module tsvc_2_s2233
  use iso_c_binding
  use omp_lib
  implicit none
contains
  subroutine tsvc_2_s2233_fp64(aa, bb, cc, LEN_2D) bind(C, name='tsvc_2_s2233_fp64')
    integer(c_int64_t), value, intent(in) :: LEN_2D
    real(c_double), dimension(0:LEN_2D-1, 0:LEN_2D-1), intent(inout) :: aa, bb
    real(c_double), dimension(0:LEN_2D-1, 0:LEN_2D-1), intent(in)    :: cc

    integer(c_int64_t) :: i, j, ii, jj, i1, i2, j1, j2, k
    integer(c_int64_t), parameter :: CHUNK = 512
    real(c_double) :: s(CHUNK)

    ! Prevent accidental nested parallelism from the auto-parallelizer.
    call omp_set_max_active_levels(1)

    ! Phase aa: aa(i,j) = aa(i,j-1) + cc(i,j), independent across i.
    ! Recurrence along j; vectorize/parallelize the contiguous i dimension.
    !$omp parallel do schedule(static) private(ii, i1, i2, j, i, k, s)
    do ii = 8, LEN_2D - 1, CHUNK
      i1 = ii
      i2 = min(ii + CHUNK - 1, LEN_2D - 1)
      do i = i1, i2
        s(i - i1 + 1) = aa(i, 7)
      end do
      do j = 8, LEN_2D - 1
        !$omp simd
        do i = i1, i2
          k = i - i1 + 1
          s(k) = s(k) + cc(i, j)
          aa(i, j) = s(k)
        end do
      end do
    end do
    !$omp end parallel do

    ! Phase bb: bb(j,i) = bb(j,i-1) + cc(j,i), independent across j.
    ! Recurrence along i; vectorize/parallelize the contiguous j dimension.
    !$omp parallel do schedule(static) private(jj, j1, j2, i, j, k, s)
    do jj = 8, LEN_2D - 1, CHUNK
      j1 = jj
      j2 = min(jj + CHUNK - 1, LEN_2D - 1)
      do j = j1, j2
        s(j - j1 + 1) = bb(j, 7)
      end do
      do i = 8, LEN_2D - 1
        !$omp simd
        do j = j1, j2
          k = j - j1 + 1
          s(k) = s(k) + cc(j, i)
          bb(j, i) = s(k)
        end do
      end do
    end do
    !$omp end parallel do
  end subroutine tsvc_2_s2233_fp64
end module tsvc_2_s2233
