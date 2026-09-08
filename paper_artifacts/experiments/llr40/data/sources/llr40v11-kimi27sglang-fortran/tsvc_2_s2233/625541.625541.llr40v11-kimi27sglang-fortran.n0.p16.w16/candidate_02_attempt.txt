module tsvc_2_s2233_mod
  use iso_c_binding, only: c_double, c_int64_t
  implicit none
contains
  subroutine tsvc_2_s2233_fp64(aa, bb, cc, LEN_2D) bind(c, name='tsvc_2_s2233_fp64')
    real(c_double), dimension(*), intent(inout) :: aa
    real(c_double), dimension(*), intent(inout) :: bb
    real(c_double), dimension(*), intent(in)    :: cc
    integer(c_int64_t), value, intent(in)       :: LEN_2D

    integer(c_int64_t) :: n, i, j, ib, ie, jb, je, idx
    integer, parameter :: BS = 256

    n = LEN_2D

    ! First recurrence: aa[j,i] = aa[j-1,i] + cc[j,i].
    ! Dependency is along j for a fixed i.  The i dimension is independent and
    ! contiguous (row-major), so tile i, run j sequentially, and vectorise i.
    !$omp parallel do schedule(static) default(none) shared(aa, cc, n) private(ib, ie, j, i, idx)
    do ib = 8, n - 1, BS
      ie = min(n - 1, ib + BS - 1)
      do j = 8, n - 1
        !$omp simd
        do i = ib, ie
          idx = j * n + i
          aa(idx + 1) = aa(idx - n + 1) + cc(idx + 1)
        end do
        !$omp end simd
      end do
    end do
    !$omp end parallel do

    ! Second recurrence: bb[i,j] = bb[i-1,j] + cc[i,j].
    ! Dependency is along i for a fixed j.  The j dimension is independent and
    ! contiguous, so tile j, run i sequentially, and vectorise j.
    !$omp parallel do schedule(static) default(none) shared(bb, cc, n) private(jb, je, i, j, idx)
    do jb = 8, n - 1, BS
      je = min(n - 1, jb + BS - 1)
      do i = 8, n - 1
        !$omp simd
        do j = jb, je
          idx = i * n + j
          bb(idx + 1) = bb(idx - n + 1) + cc(idx + 1)
        end do
        !$omp end simd
      end do
    end do
    !$omp end parallel do
  end subroutine tsvc_2_s2233_fp64
end module tsvc_2_s2233_mod
